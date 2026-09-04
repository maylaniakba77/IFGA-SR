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
├── losses.py         FGALoss (pixel + frequency [+ LPIPS]), psnr, spectral_consistency
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
    delta = self.fga(hidden_states)                                 # trainable
    return base + delta
```

The original upsampler's parameters are explicitly frozen (`requires_grad_(False)`) inside the
constructor. Because `fga.unembed` is zero-initialized, `delta == 0` until training moves it.

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
7. Validates every `--val_every` steps on a deterministic held-out split and saves the best
   checkpoint by PSNR.

### Key CLI arguments

| Flag | Default | Notes |
|---|---|---|
| `--data_dir` | *required* | Output of `cache_latents.py` |
| `--mode` | `partial` | `partial` \| `full` |
| `--inner_dim` | `64` | Lower to `32` if VRAM-constrained |
| `--iters` | `20000` | Optimizer steps (each = `--accum` micro-steps) |
| `--batch` / `--accum` | `1` / `8` | Effective batch = `batch × accum` |
| `--lr` | `1e-4` | Cosine-decayed after `--warmup` steps |
| `--amp` | `bf16` | `off` \| `fp16` \| `bf16`; prefer `bf16` on Ampere+ |
| `--w_pixel` | `1.0` | Weight of L1/L2 pixel term |
| `--w_freq` | `0.1` | Weight of frequency-domain term |
| `--w_lpips` | `0.0` | `> 0` requires the `lpips` package |
| `--freq_mode` | `full` | `full` \| `highpass` |
| `--seed` | `123456` | **Keep identical across variants** |
| `--out_dir` | *required* | Receives checkpoints, `config.json`, `history.json` |

### Loss function

`FGALoss` (`losses.py`) computes:

```
L_total = w_pixel * L_pixel + w_freq * L_frequency [+ w_lpips * L_LPIPS]
```

- `L_pixel` — L1 or L2 on the reconstruction, in `[-1, 1]` space.
- `L_frequency` — L1 on FFT magnitude. `freq_mode="full"` covers the whole spectrum;
  `freq_mode="highpass"` restricts it to frequencies above `freq_cutoff`, which targets fine
  detail specifically.
- `L_LPIPS` — optional perceptual term, disabled by default (the model is not even loaded when
  `w_lpips == 0`).

Two diagnostic metrics are also reported: `psnr` (data range 2.0, matching `[-1, 1]`) and
`spectral_consistency` (1.0 = perfect spectral match).

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
```

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

Cross each variant with sampling steps ∈ {1, 2, 3, 5} (H2) and evaluate on ImageNet-Test,
RealSRV3, and RealSet80. Use `--color_fix wavelet` when reproducing quantitative numbers on
ImageNet-Test and RealSRV3, matching the original InvSR protocol.

Report per run: PSNR, SSIM, LPIPS, spectral consistency, trainable parameter count, FGA FLOPs
(via `FGA.flops(h, w)`), and wall-clock inference latency.

---

## 9. Known Issues & Verification Checklist

### Blocking

- [ ] **`patch_decoder.py` is missing an import.** It calls `FGAUpsample2D(...)` without
      importing it, which raises `NameError` as soon as `train_fga.py` starts. Add:
      ```python
      from fga_integration.fga_upsampler import FGAUpsample2D
      ```
- [ ] **No FGA checkpoint loading in the inference path.** See §7.3. Until this exists, no
      evaluation of `partial` or `full` is meaningful.

### Verify before the full training run

- [ ] Run `cache_latents.py` with `--limit 4` and confirm the printed latent shape.
- [ ] Confirm the latent scaling convention round-trips: `cache_latents.py` stores the raw
      pipeline latent while `train_fga.py` divides by `vae.config.scaling_factor` before
      decoding. Decode one cached latent with an unmodified VAE and check it looks correct.
- [ ] Confirm the printed trainable-parameter count is non-zero and that `full` > `partial`.
- [ ] Confirm the loss at step 0 equals the baseline reconstruction loss (a consequence of
      zero-init). If it does not, zero-init is not taking effect.
- [ ] Confirm decoder gradient checkpointing actually engaged — assigning
      `vae.decoder.gradient_checkpointing = True` directly may not propagate to submodules in
      some diffusers versions. Prefer `vae._set_gradient_checkpointing(vae.decoder, True)` and
      verify by comparing peak memory.

### Minor

- `torch.cuda.amp.GradScaler` is deprecated in recent PyTorch in favour of
  `torch.amp.GradScaler("cuda")`. Harmless, but it emits warnings.
- `GradScaler` is only enabled for `--amp fp16`; under `bf16` its calls are pass-throughs, which
  is the correct behaviour.

---

## 10. Reproducibility

Every run writes `config.json` containing the complete argparse namespace, including the seed.
Archive `config.json` and `history.json` alongside each checkpoint — together they are
sufficient to reproduce and to defend the numbers reported in the thesis.

The train/validation split is deterministic (the first `val_size` sorted filenames form the
validation set), so it is stable across runs and identical between the `partial` and `full`
variants.
