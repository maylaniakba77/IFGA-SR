# FGA Integration & Training Guide

**Destination in repo:** `fga_integration/TRAINING.md`

This document describes how the **FGA (Fourier-Guided Attention) upsampler** is integrated
into the frozen InvSR pipeline, and how to reproduce the training and ablation study for the
`partial` and `full` variants.

---

## 1. Overview

InvSR performs arbitrary-step image super-resolution via diffusion inversion. Its final stage
is a **VAE decoder** that maps the sampled latent back to pixel space. That decoder is a
generic SD-Turbo autoencoder — it was never trained for super-resolution, and its nearest/conv
upsamplers are a known source of high-frequency detail loss.

FGA addresses this by attaching a lightweight, frequency-aware attention branch to the
decoder's upsampling stages. Everything else in InvSR (VAE encoder, noise predictor, SD-Turbo
U-Net) remains **completely frozen**.

### Design principles

| Principle | Implementation |
|---|---|
| **Non-destructive** | FGA is a *residual* branch: `output = original_upsampler(x) + fga(x)` |
| **Zero-init start** | `fga.unembed` weights/bias are zero-initialized, so at step 0 the model is **bit-identical to baseline InvSR** |
| **Minimal footprint** | Only FGA parameters are trainable (~0.3M); checkpoints are 1–2 MB |
| **Cheap to train** | Backbone latents are pre-cached, so one training step is just `vae.decode(...)` |
| **Backward compatible** | Default `fga_mode` is `"none"`; existing configs behave exactly as before |

---

## 2. Component Map

```
fga/archs/
├── fga_arch.py       FGA module, CAL (Correlation Attention Layer), OWXRA attention
├── subpixmlp.py      SubPixelMLP — LR→HR feature expansion used inside FGA
└── arch_util.py      MLP, trunc_normal_, conv_flops helpers

fga_integration/
├── fga_upsampler.py  FGAUpsample2D — residual wrapper around a diffusers Upsample2D
├── patch_decoder.py  inject_fga() — swaps decoder upsamplers in-place
├── cache_latents.py  Stage 0: runs frozen InvSR, dumps latents + GT to disk
├── losses.py         FGALoss (pixel + frequency + sharpness + range [+ LPIPS]),
│                     psnr, spectral_consistency, laplacian_var
├── discriminator.py  PatchDiscriminator (spectral norm) + hinge losses
└── train_fga.py      Stage 1: fine-tunes FGA only
```

### Data flow

```
LR image
   │
   ▼  (FROZEN — run once, cached to disk by cache_latents.py)
VAE Encoder → Noise Predictor → SD-Turbo U-Net
   │
   ▼  latent (4, H/8, W/8)
VAE Decoder
   ├─ up_block[0].upsamplers[0]  ──┐
   ├─ up_block[1].upsamplers[0]    │  ← FGA injected here
   ├─ up_block[2].upsamplers[0]    │    (which ones depends on `mode`)
   └─ up_block[3] (no upsampler) ──┘
   │
   ▼
HR image
```

---

## 3. The Injection Mechanism

### 3.1 `FGAUpsample2D` (`fga_upsampler.py`)

Wraps an existing diffusers `Upsample2D`:

```python
def forward(self, hidden_states, output_size=None, *args, **kwargs):
    base  = self.orig(hidden_states, output_size, *args, **kwargs)  # frozen
    if self.gain == 0.0:
        return base                                                 # FGA bypassed
    delta = self.fga(hidden_states)                                 # trainable
    return base + delta
```

…and returns `base + self.gain * delta`.

The original upsampler's parameters are explicitly frozen (`requires_grad_(False)`) inside the
constructor. Because `fga.unembed` is zero-initialized, `delta == 0` until training moves it.

#### `gain` — inference-time control of correction strength

`gain` is a plain float (not a `Parameter`, not a buffer), so it never enters the checkpoint and
never affects training, which always runs at `gain = 1.0`.

| `gain` | Effect |
|---|---|
| `0` | FGA bypassed entirely — output is **bit-identical to baseline** |
| `1` | As trained (default) |
| `< 0` | Residual inverted — acts as an unsharp mask |

Negative gain sharpens because the module, trained under distance-to-GT objectives, converges to
a **high-pass subtractor**: measured correlation between `delta` and `highpass(baseline)` is
**−0.59**. Inverting a learned smoothing operator therefore sharpens. Measured on 32 held-out
images, `partial` at `gain = -1.0` is **+80%** sharper than baseline (Laplacian variance).

Set it via `inference_invsr.py --fga_gain`, the config key `fga_gain`, or
`inject_fga(..., gain=...)`. CLI overrides YAML.

> **Reporting caveat.** At negative gain the module performs the inverse of what it was trained
> to do. Describe it as "the learned correction, applied inverted", not as "the trained model
> produces sharper output". Training it to sharpen directly (`--w_sharp`, §6) avoids the issue.

### 3.2 `inject_fga` (`patch_decoder.py`)

```python
inject_fga(vae, mode="partial", inner_dim=64) -> list[Parameter]
```

It enumerates every `(block_index, upsampler_index)` pair in `vae.decoder.up_blocks`, selects a
subset according to `mode`, replaces each with an `FGAUpsample2D`, and returns the list of
newly created trainable parameters.

| `mode` | Upsamplers patched | Purpose |
|---|---|---|
| `"none"` | none — returns `[]` immediately | Baseline / control condition |
| `"partial"` | **last one only** (`targets[-1:]`) | Cheapest variant; acts at the highest resolution |
| `"full"` | **all** upsamplers | Maximum capacity; multi-scale correction |

The returned parameter list is what `train_fga.py` hands to the optimizer. At inference time
the return value is intentionally discarded.

> **Ablation note.** `partial` vs `full` is the core of hypothesis H3: does correcting only the
> final upsampling stage capture most of the gain, or is multi-scale injection required? Report
> trainable parameter count, FLOPs, and quality metrics for both.

---

## 4. Environment

```bash
conda create -n invsr python=3.10
conda activate invsr
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[torch]"
pip install -r requirements.txt
pip install lpips        # only if training with --w_lpips > 0
```

All commands below are run **from the repository root**, not from inside `fga_integration/`.

---

## 5. Stage 0 — Cache Latents

Because the entire InvSR backbone is frozen, the latent produced for a given LR image never
changes across training iterations. Re-running the backbone every step is pure waste. Caching
turns FGA training into a cheap `latent → HR pixels` task that fits on a free T4/P100 with full
512×512 HR crops.

### Input layout

```
data/pairs/
├── lr/  <name>.png     low-resolution inputs
└── gt/  <name>.png     ground-truth HR, matching filenames
```

### Verify first (always)

```bash
python fga_integration/cache_latents.py \
  --cfg_path configs/sample-sd-turbo.yaml \
  --lr_dir data/pairs/lr --gt_dir data/pairs/gt \
  --out_dir data/cache/steps1 \
  --num_steps 1 --limit 4
```

The script asserts the returned latent shape is `(1, 4, H/8, W/8)` and aborts otherwise. This
guards against pipelines that ignore `output_type="latent"`. Only after this passes should you
drop `--limit` and process the full dataset.

### Output layout

```
data/cache/steps1/
├── latent/ <name>.npy   (4, h, w)   float16
└── gt/     <name>.npy   (3, H, W)   float16, range [0, 1]
```

Storage cost is negligible: a 64×64×4 float16 latent is ~32 KB, so 20,000 images ≈ 640 MB.
The script skips already-cached files, so it is safe to resume after an interruption.

Repeat per sampling-step regime you intend to study (`--num_steps 1 / 2 / 3 / 5`, hypothesis H2).
**One cache directory is shared by both the `partial` and `full` runs** — do not regenerate it.

---

## 6. Stage 1 — Train FGA

### Commands

```bash
# Variant A — partial
python fga_integration/train_fga.py \
  --data_dir data/cache/steps1 \
  --mode partial \
  --iters 20000 --batch 1 --accum 8 \
  --out_dir experiments/fga_partial

# Variant B — full
python fga_integration/train_fga.py \
  --data_dir data/cache/steps1 \
  --mode full \
  --iters 20000 --batch 1 --accum 8 \
  --out_dir experiments/fga_full
```

Both variants **must** use identical `--data_dir`, `--iters`, `--seed`, `--lr`, and loss weights.
The only difference permitted between them is `--mode`. Otherwise the ablation is not a
controlled comparison.

### What the script does

1. Loads **only** `AutoencoderKL` (float32) — no U-Net, no noise predictor.
2. Freezes the entire VAE, then calls `inject_fga` and re-enables grad on FGA parameters only.
3. Prints trainable parameter count and its percentage of the VAE — **record this for your
   results table**.
4. Enables decoder gradient checkpointing to reduce peak memory.
5. Trains with AdamW, linear warmup + cosine decay, gradient clipping, and mixed precision via
   `autocast` (weights stay float32 — FGA contains LayerNorm and softmax attention that are
   unstable in pure fp16).
6. Computes the loss in float32 because FFT is unstable at half precision.
7. Validates every `--val_every` steps on a held-out split and saves the best checkpoint by
   `--select_by`.

### Key CLI arguments

| Flag | Default | Notes |
|---|---|---|
| `--data_dir` | *required* | Output of `cache_latents.py` |
| `--mode` | `partial` | `partial` \| `full` |
| `--inner_dim` | `64` | Lower to `32` if VRAM-constrained |
| `--iters` | `20000` | Optimizer steps (each = `--accum` micro-steps) |
| `--batch` / `--accum` | `1` / `8` | Effective batch = `batch × accum` |
| `--crop` | `0` | Random crop in HR pixels (multiple of 8). Makes `--batch > 1` legal |
| `--lr` | `1e-4` | Cosine-decayed after `--warmup` steps |
| `--amp` | `bf16` | `off` \| `fp16` \| `bf16`; prefer `bf16` on Ampere+ |
| **Split** | | |
| `--split_by` | `scene` | `scene` holds whole scenes and is **stable under `--draws`**; `name` is the legacy behaviour |
| `--val_scenes` | `8` | Scenes held out when `--split_by scene` |
| `--val_size` | `32` | Files held out when `--split_by name` (legacy only) |
| **Loss** | | |
| `--w_pixel` | `1.0` | L1/L2 pixel term |
| `--w_freq` | `0.1` | Frequency term |
| `--freq_mode` | `full` | `full` \| `highpass` \| `magnitude` |
| `--freq_cutoff` | `0.25` | Band limit, as a fraction of Nyquist |
| `--w_lpips` | `0.0` | `> 0` requires the `lpips` package |
| `--lpips_net` | `alex` | `alex` \| `vgg` — **prefer `vgg`**, see below |
| `--w_sharp` | `0.0` | One-sided sharpness term — the only term that can demand output **sharper than baseline** |
| `--sharp_ratio` | `1.0` | Sharpness target as a multiple of GT |
| `--w_range` | `1.0` | Penalty for pixels outside `[-1, 1]`. **Keep > 0 whenever `--w_sharp > 0`** |
| **Adversarial** | | |
| `--w_gan` | `0.0` | `0` = no discriminator is built. Useful range for SR: 0.02–1.0 |
| `--d_lr` / `--d_base` | `1e-4` / `64` | Discriminator LR and base width |
| `--gan_start` | `1000` | Steps before the adversarial signal is enabled |
| **Selection** | | |
| `--select_by` | `psnr` | `psnr` \| `lpips` \| `loss` \| `sharp` |
| `--seed` | `123456` | **Keep identical across variants** |
| `--out_dir` | *required* | Receives checkpoints, `config.json`, `history.json` |

### Loss function

`FGALoss` (`losses.py`) computes:

```
L_total = w_pixel·L_pixel + w_freq·L_freq [+ w_lpips·L_LPIPS]
                          [+ w_sharp·L_sharp] [+ w_range·L_range] [+ w_gan·L_GAN]
```

**`L_pixel`** — L1 or L2 on the reconstruction, in `[-1, 1]` space.

**`L_freq`** — three variants, and the distinction matters more than it looks:

| `freq_mode` | What it measures | Gradient direction |
|---|---|---|
| `full` | L1 on the **complex** rFFT difference, whole spectrum | **Toward smoothing** |
| `highpass` | Same, restricted to the high band | **Toward smoothing** |
| `magnitude` | L1 on the **magnitude** difference, high band | Toward matching detail energy |

`full` and `highpass` use an orthonormal basis, so by Parseval they are equivalent to a pixel
loss in a rotated basis — not an independent supervision signal. Both are phase-sensitive:
texture that is statistically correct but shifted a few pixels is punished as hard as a pixel
loss punishes it, so the cheapest way to reduce them is to **attenuate** that texture.
`magnitude` discards phase first, so shifted texture is not punished.

**`L_sharp`** (`--w_sharp`) — `relu(ratio − lap_energy(pred) / lap_energy(gt))`. One-sided:
being too sharp is not punished at all. Smoothing *raises* it, so smoothing stops being an
escape route. `sharp_ratio` is your choice, not a property of the data — `1.0` matches the real
photo, `> 1.0` exceeds it.

**`L_range`** (`--w_range`) — `relu(|pred| − 1)`. Mandatory whenever `w_sharp > 0`: the sharpness
term is unbounded and computed on unclamped output, so the optimizer can buy Laplacian energy for
free by pushing pixels out of range. Those clip per-channel at inference and appear as saturated
magenta speckles. Clamping `pred` instead does **not** work — the gradient outside the range
becomes zero, so the spikes are never punished.

**`L_GAN`** (`--w_gan`) — hinge loss against a spectral-norm PatchGAN (`discriminator.py`). This
is the only term that asks a distribution question ("does this texture look real") rather than a
distance question ("how close is this to GT"), and therefore the only one that escapes the
distortion–perception trade-off. See §10.

**`--lpips_net`.** LPIPS-alex and LPIPS-VGG were measured moving in **opposite directions** on the
same checkpoints: alex improved while VGG worsened. Alex is the less blur-sensitive of the two, so
optimizing it can be satisfied by smoothing. Use `vgg`.

Diagnostic metrics reported at validation: `psnr`, `spectral_consistency`, and `lap_ratio`
(Laplacian variance relative to GT; `1.0` = GT sharpness).

> **`--select_by` is not a cosmetic choice.** `psnr`, `lpips`, and `loss` **all reward
> smoothing** — under a fidelity objective they consistently select the blurriest checkpoint among
> the validation points. `sharp` selects the checkpoint whose `lap_ratio` is closest to the target
> (`--sharp_ratio` when `--w_sharp > 0`, otherwise `1.0`).

### Outputs

```
experiments/fga_partial/
├── config.json               full argparse snapshot — cite this in the thesis
├── history.json              per-step training + validation log
├── fga_partial_best.pth      best validation PSNR ← use this for evaluation
├── fga_partial_last.pth      periodic checkpoint
└── fga_partial_final.pth     end of training
```

Checkpoints contain **only** keys matching `".fga."`, plus `mode`, `inner_dim`, `step`, and
`metrics`. File size is ~1–2 MB.

---

## 7. Stage 2 — Inference Integration

### 7.1 Sampler hook

`sampler_invsr.py` wires FGA into the full pipeline at the end of `BaseSampler.build_model()`:

```python
from fga_integration.patch_decoder import inject_fga
...
self.sd_pipe = sd_pipe
inject_fga(sd_pipe.vae, mode=self.configs.get("fga_mode", "none"))
```

`self.configs.get("fga_mode", "none")` reads the key from the YAML config and defaults to
`"none"`, so every pre-existing config keeps running as plain baseline InvSR.

### 7.2 Config keys

```yaml
fga_mode: partial                                        # none | partial | full
fga_ckpt: experiments/fga_partial/fga_partial_best.pth
fga_gain: 1.0                                            # see §3.1; CLI overrides YAML
```

`inference_invsr.py` exposes `--fga_mode`, `--fga_ckpt`, and `--fga_gain`, so one config serves
baseline and every variant.

> `inference_invsr.py` **overwrites** `cache_dir` and `model_start.ckpt_path` from the YAML with
> its own defaults (`./weights`). Pass `--sd_path` and `--started_ckpt_path` explicitly, or it
> re-downloads sd-turbo on every run.

### 7.3 Loading trained weights — REQUIRED

> **Critical:** `inject_fga` only installs the *architecture*. Because `unembed` is
> zero-initialized, `delta == 0` and the output is **identical to baseline** until trained
> weights are loaded. Without the block below, your `partial` and `full` evaluations will be
> indistinguishable from the control condition.

Add immediately after the `inject_fga` call:

```python
fga_mode = self.configs.get("fga_mode", "none")
if fga_mode != "none":
    ckpt = torch.load(self.configs.fga_ckpt, map_location="cuda")
    assert ckpt["mode"] == fga_mode, (
        f"Checkpoint was trained with mode={ckpt['mode']} "
        f"but config requests fga_mode={fga_mode}"
    )
    missing, unexpected = sd_pipe.vae.load_state_dict(ckpt["state_dict"], strict=False)
    assert not unexpected, f"Unexpected keys in FGA checkpoint: {unexpected}"
    self.write_log(f"Loaded FGA weights ({fga_mode}) from {self.configs.fga_ckpt}")
```

`strict=False` is necessary because the checkpoint holds only `.fga.` keys, not the whole VAE.
The `unexpected` assertion is what actually verifies the load succeeded — a silent no-op here is
the single easiest way to invalidate an entire experiment.

### 7.4 Precision caveat

If the config enables `vae_fp16`, `inject_fga` casts FGA to fp16 to match the original
upsampler's dtype. FGA weights are trained in float32 precisely because its softmax attention
and LayerNorm are unstable in half precision. For evaluation runs, either disable `vae_fp16`,
or explicitly measure whether fp16 degrades your metrics and report it.

---

## 8. Experiment Matrix

| Run | `fga_mode` | Trainable params | Purpose |
|---|---|---|---|
| Baseline | `none` | 0 | Control — unmodified InvSR |
| FGA-partial | `partial` | ~0.3M | H3: is last-stage correction sufficient? |
| FGA-full | `full` | ~0.3M × N | H3: does multi-scale injection add value? |

Cross each variant with sampling steps ∈ {1, 3, 5} (H2) and evaluate on the three datasets below
— 27 combinations. Use `--color_fix wavelet` when reproducing quantitative numbers on
ImageNet-Test and RealSRV3, matching the original InvSR protocol.

### Datasets (per the thesis protocol, §2.4.3)

| Role | Dataset | Notes |
|---|---|---|
| Training | **LSDIR subset + 20,000 FFHQ faces** | Real-ESRGAN degradation, official InvSR protocol |
| Eval (reference) | **ImageNet-Test** | 3,000 synthetic, LR 128 → HR 512, from the InvSR repo |
| Eval (reference) | **RealSRV3** | 100 real-world pairs with genuine HR ground truth |
| Eval (no-reference) | **RealSet80** | 80 real-world images, no GT — already in `testdata/RealSet80` |

`modal_train.py::prepare --source {lsdir,ffhq,div2k_valid,div2k_train}` fetches training data;
LSDIR and FFHQ stream from HuggingFace so a subset does not require the full download. Use
`--data_tag` throughout the pipeline to keep datasets in separate directories.

DIV2K is **not** part of the thesis protocol — it is a fast development path only. Results
trained or evaluated on DIV2K do not answer the experiment matrix above.

Report per run: PSNR, SSIM, LPIPS, spectral consistency, Laplacian variance relative to GT,
CLIPIQA/MUSIQ on RealSet80, trainable parameter count, FGA FLOPs (via `FGA.flops(h, w)`), and
wall-clock inference latency.

---

## 9. Known Issues & Verification Checklist

### Fixed and committed

Both items previously listed as blocking are resolved: `patch_decoder.py` imports
`FGAUpsample2D`, and `sampler_invsr.py` loads `fga_ckpt` with asserts on `mode`, on unexpected
keys, and on unpopulated `.fga.` keys.

### Defects found during experiments — read before trusting old results

**Duplicate degradation draws (`make_pairs.py`).** The resume path `continue`d over existing
files without calling `ds[i]`, so the RNG never advanced. Re-running with a larger `--draws`
therefore **replayed the same kernels**: `0801_d0.png` and `0801_d4.png` were byte-identical
(same MD5). A dataset believed to hold 800 pairs actually held 400 unique ones. Fixed by seeding
per `(stem, draw, seed)` via md5, so each draw is deterministic, distinct, and independent of
which files were skipped.

**Validation split shifting with `--draws` (legacy `--split_by name`).** Files are named
`<scene>_d<draw>`, so selecting the *first 32 names* means `draws=4` yields 8 scenes × 4 draws
while `draws=8` yields **4 scenes × 8 draws**. Only 16 names overlap, and scenes move from
validation into training. Checkpoints trained either side of a `--draws` change are not
comparable. Fixed by `--split_by scene` (now the default), which holds whole scenes.

**Out-of-range pixel spikes under `--w_sharp`.** See §6, `L_range`. Symptom: scattered saturated
magenta pixels (measured mean RGB `[231, 58, 137]`). Fixed by `--w_range`, which cut new
out-of-range pixels from 0.104% to 0.0115%.

### Verify before the full training run

- [ ] `cache_latents.py --limit 4` prints the expected latent shape.
- [ ] Latent scaling round-trips: `cache_latents.py` stores the **raw** pipeline latent while
      `train_fga.py` divides by `vae.config.scaling_factor`. Decode one cached latent with an
      unmodified VAE and confirm it looks plausible.
- [ ] Trainable-parameter count is non-zero and `full` > `partial`.
- [ ] Zero-init holds: with untrained weights, `max|rec − baseline|` must be **exactly 0.0**,
      not merely small. `modal_train.py::gate2` checks all four criteria automatically.
- [ ] Output actually differs from baseline after loading weights (`gate_diff`). A value of
      `0.0` means the weights never reached the model and every downstream metric is a duplicate
      of baseline — a wiring bug that reads as "FGA makes no difference".
- [ ] Decoder gradient checkpointing engaged — assigning `vae.decoder.gradient_checkpointing`
      may not propagate to submodules in some diffusers versions.

### Measurement pitfalls

**Sharpness must be measured as local contrast, not spectral energy share.** Baseline InvSR
measured a *higher* high-frequency energy fraction than GT (0.5755 vs 0.5488) while its edge
contrast was 29% *lower* — its HF content is diffuse low-amplitude noise, not structured edges.
Laplacian variance is not fooled by this; the HF fraction is.

**Scene count dominates the sharpness metric.** GT Laplacian variance measured **2.8× apart**
between an 8-scene and a 4-scene subset. On one subset baseline sat 29% *below* GT; on the other,
46% *above*. Evaluate on ~20 scenes minimum, and never compare conditions measured on different
image sets.

### Minor

- `torch.cuda.amp.GradScaler` is deprecated in favour of `torch.amp.GradScaler("cuda")`.
  Harmless warnings only.
- `GradScaler` is enabled only for `--amp fp16`; under `bf16` its calls are pass-throughs.

---

## 10. Measured Findings

All figures below come from held-out scenes, with every condition evaluated on identical images.

**The module converges to a smoothing operator under distance objectives.** Correlation between
the learned `delta` and `highpass(baseline)` is **−0.59** — it learned to *subtract* high
frequencies. The cause is structural: for ×4 SR from heavy degradation the posterior is wide, and
the distance-minimizing solution is the posterior **mean**, which is blurry by definition.
Baseline InvSR is a posterior **sample** and is therefore sharper.

**The frequency loss determines the outcome more than its weight does.** Sharpness relative to GT:

| `freq_mode` | Config | Sharpness vs GT |
|---|---|---|
| `full` | `w_freq 0.1`, `select_by psnr` | **0.04×** |
| `magnitude` | `w_freq 1.0` | 0.63× |
| `magnitude` | `+ vgg`, `w_pixel 0.1` | 0.69× |

**Rescaling the residual cannot escape the trade-off.** Sweeping `gain` from 0 to 1: every
reference metric improves monotonically toward `gain = 1`, every no-reference metric degrades
monotonically. No interior optimum exists — rescaling only slides along the trade-off curve.

**The adversarial term is what moves it.** Converged sharpness rose from ~0.37× to ~0.70× GT as
`w_gan` increased and `w_lpips` decreased. With `--w_sharp` added, `full` reached **1.52× GT —
+38% sharper than baseline at `gain = 1.0`**, from training alone.

**H3 answered from two directions.** Under fidelity objectives `partial ≈ full` on every metric
(PSNR 24.36 vs 24.32). Under a sharpness objective they diverge sharply — **0.96× vs 1.52× GT**.
Last-stage correction suffices for fidelity; multi-scale injection is required for sharpening.

**More sampling steps did not help the baseline.** At 3 steps, baseline InvSR was worse than at 1
step on all six metrics.

---

## 11. Reproducibility

Every run writes `config.json` containing the complete argparse namespace, including the seed and
the split configuration. Archive `config.json` and `history.json` alongside each checkpoint.

The split is deterministic. With `--split_by scene` (default) the first `--val_scenes` sorted
scenes and **all their draws** form the validation set, which makes it stable under changes to
`--draws` — unlike the legacy `name` split. Verified on the real file list: zero file overlap and
zero scene overlap between training and evaluation.

> When comparing checkpoints trained under different splits, restrict evaluation to scenes held
> out from **every** checkpoint being compared.
