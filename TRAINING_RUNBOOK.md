# FGA Training Runbook — Step by Step

**Destination in repo:** `fga_integration/TRAINING_RUNBOOK.md`
**Companion document:** `fga_integration/TRAINING.md` (architecture and API reference)

This is the operational procedure for taking the FGA module from a fresh checkout to two
trained checkpoints (`partial` and `full`) ready for evaluation.

All commands are run **from the repository root**.

---

## Contents

- [Phase 0 — Unblock the codebase](#phase-0--unblock-the-codebase)
- [Phase 1 — Environment](#phase-1--environment)
- [Phase 2 — Prepare paired data](#phase-2--prepare-paired-data)
- [Phase 3 — GATE 1: Caching smoke test](#phase-3--gate-1-caching-smoke-test)
- [Phase 4 — Full latent caching](#phase-4--full-latent-caching)
- [Phase 5 — GATE 2: Training smoke test](#phase-5--gate-2-training-smoke-test)
- [Phase 6 — Train the `partial` variant](#phase-6--train-the-partial-variant)
- [Phase 7 — Train the `full` variant](#phase-7--train-the-full-variant)
- [Phase 8 — GATE 3: Inference wiring](#phase-8--gate-3-inference-wiring)
- [Phase 9 — Evaluation matrix](#phase-9--evaluation-matrix)
- [Troubleshooting](#troubleshooting)
- [Progress checklist](#progress-checklist)

---

## Phase 0 — Unblock the codebase

Two defects currently prevent training from starting at all. Fix both before anything else.

### 0a. Add the missing import

`fga_integration/patch_decoder.py` calls `FGAUpsample2D(...)` but never imports it. Add at the
top of the file:

```python
from fga_integration.fga_upsampler import FGAUpsample2D
```

Without this, `train_fga.py` raises `NameError` immediately on startup.

### 0b. Ensure the packages are importable

Create empty `__init__.py` files if they do not already exist:

```bash
touch fga/__init__.py fga/archs/__init__.py fga_integration/__init__.py
```

### 0c. Verify

```bash
python -c "from fga_integration.patch_decoder import inject_fga; print('OK')"
```

**Expected:** `OK`. Do not proceed until this passes.

---

## Phase 1 — Environment

```bash
conda create -n invsr python=3.10
conda activate invsr
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[torch]"
pip install -r requirements.txt
pip install lpips        # only if you intend to train with --w_lpips > 0
```

Download the finetuned LPIPS weights into `weights/` as described in the repository README if
you plan to use the perceptual loss term.

### Record your hardware

Note the GPU model and VRAM now — it determines your AMP mode and `--inner_dim` budget.

| GPU generation | Recommended `--amp` | Reason |
|---|---|---|
| Ampere and newer (A100, RTX 30xx/40xx, L4) | `bf16` | Native bf16 support; no loss scaling needed |
| Turing and older (T4, RTX 20xx, P100) | `fp16` | bf16 is not natively supported |
| Debugging / numerical issues | `off` | Slowest but fully deterministic in float32 |

---

## Phase 2 — Prepare paired data

FGA training requires LR↔GT pairs with **identical filenames** in two directories:

```
data/pairs/
├── lr/  <name>.png
└── gt/  <name>.png
```

Choose one of two sourcing strategies:

**Synthetic (recommended).** Take an HR dataset (DIV2K, LSDIR) and generate the LR inputs with
the Real-ESRGAN degradation pipeline already vendored in `basicsr/`. This matches the original
InvSR training protocol, which makes the comparison against baseline defensible.

**Real.** Use genuine LR/HR pairs from RealSRV3.

> **Start small.** Use **200–500 pairs** for the first pass through this runbook. Only scale to
> the full dataset after Phase 5 has passed and you have measured the actual cost per step.

---

## Phase 3 — GATE 1: Caching smoke test

```bash
python fga_integration/cache_latents.py \
  --cfg_path configs/sample-sd-turbo.yaml \
  --lr_dir data/pairs/lr --gt_dir data/pairs/gt \
  --out_dir data/cache/steps1 \
  --num_steps 1 --limit 4
```

### Pass criteria

1. The log prints `[cache] bentuk latent = (1, 4, H/8, W/8)`. The script raises `RuntimeError`
   if the shape is wrong, which indicates the pipeline is ignoring `output_type="latent"`.
2. Four `.npy` files appear in both `data/cache/steps1/latent/` and `data/cache/steps1/gt/`.

### Additional check — latent scaling round-trip

This check is strongly recommended and takes two minutes. `cache_latents.py` stores the raw
pipeline latent, while `train_fga.py` divides by `vae.config.scaling_factor` before decoding.
If that convention is mismatched, training will silently learn to compensate for a bug instead
of improving detail — and the resulting numbers will be meaningless.

```python
import numpy as np, torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image
from pathlib import Path

vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae",
                                    torch_dtype=torch.float32).cuda().eval()
p = sorted(Path("data/cache/steps1/latent").glob("*.npy"))[0]
lat = torch.from_numpy(np.load(p).astype("float32"))[None].cuda()

with torch.no_grad():
    rec = vae.decode(lat / vae.config.scaling_factor).sample

save_image((rec.clamp(-1, 1) + 1) / 2, "roundtrip_check.png")
print("latent", tuple(lat.shape), "-> recon", tuple(rec.shape))
```

**Pass criterion:** `roundtrip_check.png` is a plausible image. If it is grey noise or heavily
distorted, stop and resolve the scaling convention before continuing.

**Do not proceed past this gate until both checks pass.**

---

## Phase 4 — Full latent caching

```bash
python fga_integration/cache_latents.py \
  --cfg_path configs/sample-sd-turbo.yaml \
  --lr_dir data/pairs/lr --gt_dir data/pairs/gt \
  --out_dir data/cache/steps1 --num_steps 1
```

Notes:

- The script skips files that are already cached, so it is safe to interrupt and resume.
- Storage is cheap: a 64×64×4 float16 latent is ~32 KB, so 20,000 images ≈ 640 MB.
- **This single cache directory is shared by both the `partial` and `full` runs.** Never
  regenerate it between variants — a shared cache is part of what makes the ablation controlled.
- Repeat with `--num_steps 2 / 3 / 5` into separate output directories only if you are studying
  the sampling-step regimes (hypothesis H2).

---

## Phase 5 — GATE 2: Training smoke test

Run a deliberately tiny job before committing to the real one.

```bash
python fga_integration/train_fga.py \
  --data_dir data/cache/steps1 \
  --mode partial \
  --iters 50 --accum 2 \
  --log_every 5 --val_every 25 \
  --out_dir experiments/smoke_partial
```

### Pass criteria

**1. Trainable parameter count is non-zero.**

Look for `[model] mode=partial | parameter dilatih = N (X% dari VAE)`. If `N == 0`, `inject_fga`
found no upsamplers to patch and nothing is being trained.

**2. Step-0 loss equals the baseline reconstruction loss.**

This is a direct consequence of the zero-initialized `unembed`: at iteration 0 the model output
is bit-identical to baseline InvSR. If the initial loss is unexpectedly large or diverges
immediately, zero-init is not taking effect and the residual branch is corrupting the pretrained
prior from the very first step.

**3. Measure the real cost per step, then extrapolate.**

Read the elapsed-minutes figure in the log and compute the cost per micro-step. A default full
run is `--iters 20000 × --accum 8 = 160,000` micro-steps.

```
projected_hours = (minutes_elapsed / micro_steps_done) * 160000 / 60
```

**Do this arithmetic before starting Phase 6.** On a free-tier T4 at 512×512 the projection can
run into multiple days. If the number is unacceptable, adjust now:

| Lever | Effect |
|---|---|
| `--inner_dim 32` | Roughly halves FGA compute and parameters |
| `--iters 10000` | Halves wall-clock; apply to **both** variants |
| `--accum 4` | Smaller effective batch, faster steps, noisier gradients |
| Smaller HR crops | Largest single saving; requires re-caching |

**4. Repeat with `--mode full`.**

```bash
python fga_integration/train_fga.py \
  --data_dir data/cache/steps1 --mode full \
  --iters 50 --accum 2 --log_every 5 --val_every 25 \
  --out_dir experiments/smoke_full
```

Confirm the trainable parameter count is **strictly greater** than the `partial` run. If the two
are equal, `inject_fga`'s mode selection is not working and your two experimental conditions are
in fact identical.

**5. Record both parameter counts** — they belong in your results table.

Then clean up:

```bash
rm -rf experiments/smoke_partial experiments/smoke_full
```

---

## Phase 6 — Train the `partial` variant

```bash
python fga_integration/train_fga.py \
  --data_dir data/cache/steps1 \
  --mode partial \
  --iters 20000 --batch 1 --accum 8 \
  --lr 1e-4 --amp bf16 --seed 123456 \
  --w_pixel 1.0 --w_freq 0.1 --freq_mode full \
  --out_dir experiments/fga_partial
```

Substitute `--amp fp16` on Turing-class GPUs (see Phase 1).

### Monitoring

- Console prints every `--log_every` steps: learning rate and each loss component.
- Validation every `--val_every` steps prints `VAL loss`, `psnr` (dB), and `spec`.
- The best-PSNR checkpoint is saved automatically as `fga_partial_best.pth`.
- `history.json` is rewritten every `--save_every` steps — safe to inspect mid-run.

### What healthy training looks like

- Total loss decreases from the baseline value; the frequency term should fall alongside pixel
  loss rather than trading off against it.
- Validation PSNR improves over baseline and then plateaus.
- `spec` (spectral consistency, 1.0 = perfect) trends upward.

If validation PSNR sits exactly at baseline for thousands of steps, the module is learning to
output zero — meaning the residual branch is not finding usable signal. Check the learning rate
and confirm gradients are actually reaching FGA parameters.

### Outputs

```
experiments/fga_partial/
├── config.json               argparse snapshot — archive this
├── history.json              training + validation log
├── fga_partial_best.pth      ← use this for evaluation
├── fga_partial_last.pth
└── fga_partial_final.pth
```

Checkpoints contain only `.fga.` keys (~1–2 MB).

---

## Phase 7 — Train the `full` variant

Identical command with two changes only: `--mode` and `--out_dir`.

```bash
python fga_integration/train_fga.py \
  --data_dir data/cache/steps1 \
  --mode full \
  --iters 20000 --batch 1 --accum 8 \
  --lr 1e-4 --amp bf16 --seed 123456 \
  --w_pixel 1.0 --w_freq 0.1 --freq_mode full \
  --out_dir experiments/fga_full
```

> **Controlled-comparison requirement.** `--data_dir`, `--iters`, `--batch`, `--accum`, `--lr`,
> `--seed`, `--amp`, and every loss weight **must be byte-identical** to Phase 6. If any one of
> them differs, the `partial` vs `full` comparison is confounded and no conclusion about
> hypothesis H3 can be drawn from it.

Verify afterwards by diffing the two configs — every key except `mode` and `out_dir` should match:

```bash
diff <(python -m json.tool experiments/fga_partial/config.json) \
     <(python -m json.tool experiments/fga_full/config.json)
```

---

## Phase 8 — GATE 3: Inference wiring

### 8a. Add checkpoint loading

`sampler_invsr.py` already installs the architecture at the end of `build_model()`:

```python
inject_fga(sd_pipe.vae, mode=self.configs.get("fga_mode", "none"))
```

This alone does nothing observable, because `unembed` is zero-initialized. Add immediately after:

```python
fga_mode = self.configs.get("fga_mode", "none")
if fga_mode != "none":
    ckpt = torch.load(self.configs.fga_ckpt, map_location="cuda")
    assert ckpt["mode"] == fga_mode, (
        f"Checkpoint mode={ckpt['mode']} does not match config fga_mode={fga_mode}"
    )
    missing, unexpected = sd_pipe.vae.load_state_dict(ckpt["state_dict"], strict=False)
    assert not unexpected, f"Unexpected keys in FGA checkpoint: {unexpected}"
    self.write_log(f"Loaded FGA weights ({fga_mode}) from {self.configs.fga_ckpt}")
```

`strict=False` is required because the checkpoint holds only `.fga.` keys. The `unexpected`
assertion is what actually proves the load succeeded.

### 8b. Add config keys

```yaml
fga_mode: partial                                        # none | partial | full
fga_ckpt: experiments/fga_partial/fga_partial_best.pth
```

Configs without `fga_mode` default to `"none"` and behave as unmodified baseline InvSR.

### 8c. Verify

Run inference on the same input twice — once with `fga_mode: none`, once with `fga_mode: partial`
— and compare the outputs.

```bash
python -c "
from PIL import Image; import numpy as np
a = np.asarray(Image.open('out_baseline/img.png')).astype(float)
b = np.asarray(Image.open('out_partial/img.png')).astype(float)
print('mean abs diff:', np.abs(a - b).mean())
"
```

**Pass criterion:** the difference is clearly non-zero. If it is exactly `0.0`, the checkpoint
did not load and every downstream number would be a duplicate of the baseline.

### 8d. Precision caveat

If your config enables `vae_fp16`, `inject_fga` casts FGA to fp16 to match the wrapped
upsampler. FGA is trained in float32 because its softmax attention and LayerNorm are unstable at
half precision. For reported results, either disable `vae_fp16` during evaluation, or measure
the fp16 degradation explicitly and state it.

---

## Phase 9 — Evaluation matrix

| Condition | `fga_mode` | `fga_ckpt` |
|---|---|---|
| Baseline (control) | `none` | — |
| FGA-partial | `partial` | `experiments/fga_partial/fga_partial_best.pth` |
| FGA-full | `full` | `experiments/fga_full/fga_full_best.pth` |

Cross all three conditions with sampling steps ∈ {1, 2, 3, 5} (hypothesis H2), on:

- ImageNet-Test (synthetic) — add `--color_fix wavelet`
- RealSRV3 (real pairs) — add `--color_fix wavelet`
- RealSet80 (real, no ground truth — no-reference metrics only)

Report per cell: PSNR, SSIM, LPIPS, spectral consistency, trainable parameter count, FGA FLOPs
(via `FGA.flops(h, w)`), and wall-clock inference latency.

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `NameError: FGAUpsample2D` | Phase 0a not applied | Add the import |
| `ModuleNotFoundError: fga_integration` | Running from wrong directory, or missing `__init__.py` | Run from repo root; apply Phase 0b |
| `FileNotFoundError: Tidak ada latent di ...` | Cache directory empty or wrong path | Re-check `--data_dir` matches `cache_latents.py --out_dir` |
| `RuntimeError` on latent shape | Pipeline ignores `output_type="latent"` | Inspect the pipeline's return type before caching |
| `parameter dilatih = 0` | `inject_fga` found no upsamplers | Print `vae.decoder.up_blocks` and inspect `.upsamplers` |
| `partial` and `full` param counts equal | Mode selection not applied | Check the `targets[-1:]` slice in `patch_decoder.py` |
| Loss is `nan` immediately | fp16 instability in attention/FFT | Switch to `--amp bf16`, or `--amp off` to isolate |
| Validation PSNR pinned at baseline | Gradients not reaching FGA | Confirm `requires_grad` is `True` on FGA params only |
| OOM during training | HR crop or `inner_dim` too large | `--inner_dim 32`, verify gradient checkpointing engaged |
| Inference identical to baseline | Checkpoint not loaded | Apply Phase 8a; check the `unexpected` assertion |

### Verify gradient checkpointing actually engaged

`train_fga.py` sets `vae.decoder.gradient_checkpointing = True` by direct attribute assignment,
which may not propagate to submodules in some diffusers versions. If you hit OOM, prefer:

```python
vae._set_gradient_checkpointing(vae.decoder, True)
```

Confirm the change took effect by comparing `torch.cuda.max_memory_allocated()` before and after.

---

## Progress checklist

**Phase 0 — Unblock**
- [ ] Import added to `patch_decoder.py`
- [ ] `__init__.py` files present
- [ ] `inject_fga` imports cleanly

**Phase 1–2 — Setup**
- [ ] Environment created, dependencies installed
- [ ] GPU generation recorded, AMP mode chosen
- [ ] LR/GT pairs prepared with matching filenames

**Phase 3–4 — Caching**
- [ ] GATE 1: latent shape verified with `--limit 4`
- [ ] GATE 1: latent scaling round-trip image looks correct
- [ ] Full cache generated

**Phase 5 — Smoke test**
- [ ] GATE 2: trainable parameter count non-zero
- [ ] GATE 2: step-0 loss matches baseline
- [ ] GATE 2: runtime extrapolated and accepted
- [ ] GATE 2: `full` param count > `partial` param count
- [ ] Both parameter counts recorded for the results table

**Phase 6–7 — Training**
- [ ] `partial` variant trained; `fga_partial_best.pth` saved
- [ ] `full` variant trained; `fga_full_best.pth` saved
- [ ] `config.json` diff confirms only `mode` and `out_dir` differ

**Phase 8–9 — Evaluation**
- [ ] Checkpoint loading added to `sampler_invsr.py`
- [ ] GATE 3: output differs measurably from baseline
- [ ] Precision (fp16 vs fp32) decision made and documented
- [ ] Full evaluation matrix completed
- [ ] `config.json` + `history.json` archived with every checkpoint
