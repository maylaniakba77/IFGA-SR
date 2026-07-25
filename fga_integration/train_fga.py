"""
train_fga.py — Fine-tuning ringan modul FGA pada blok upsampling decoder VAE InvSR.

RUANG LINGKUP YANG DILATIH
    Hanya parameter modul FGA (kurang lebih 0,3 juta parameter).
    Encoder VAE, Noise Predictor, dan U-Net SD-Turbo TIDAK dimuat sama sekali di
    skrip ini — latent keluarannya sudah di-cache lebih dulu oleh cache_latents.py.
    Yang dimuat hanya AutoencoderKL, dan seluruh bobot bawaannya dibekukan.

    Konsekuensinya: satu langkah pelatihan hanya menjalankan `vae.decode(...)`,
    sehingga muat nyaman pada GPU T4/P100 gratis dengan crop HR 512x512 penuh.

CARA PAKAI
    python train_fga.py \
        --data_dir  data/cache/steps1 \
        --mode      partial \
        --iters     20000 \
        --batch     1 \
        --accum     8 \
        --out_dir   experiments/fga_partial

    Ulangi dengan --mode full untuk varian kedua (studi ablasi H3).

CATATAN DESAIN
    Modul FGA diinisialisasi zero pada lapisan `unembed`, sehingga pada iterasi
    ke-0 keluaran model IDENTIK dengan baseline InvSR. Pelatihan hanya bergerak
    menjauh dari baseline bila memang menurunkan loss — membuat proses stabil dan
    tidak merusak prior pretrained.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from diffusers import AutoencoderKL

from fga_integration.losses import FGALoss, psnr, spectral_consistency
from fga_integration.patch_decoder import inject_fga


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class LatentHRDataset(Dataset):
    """Pasangan (latent hasil InvSR, ground-truth HR) yang sudah di-cache.

    Struktur direktori yang diharapkan (dihasilkan oleh cache_latents.py):
        data_dir/
        ├── latent/  <nama>.npy   (4, h, w)  float16
        └── gt/      <nama>.npy   (3, H, W)  float16, rentang [0, 1]
    """

    def __init__(self, data_dir: str, split: str = "train", val_size: int = 32):
        root = Path(data_dir)
        names = sorted(p.stem for p in (root / "latent").glob("*.npy"))
        if len(names) == 0:
            raise FileNotFoundError(f"Tidak ada latent di {root / 'latent'}")

        # Split deterministik agar train/val konsisten antar-run dan antar-varian
        val_size = min(val_size, max(1, len(names) // 10))
        self.names = names[val_size:] if split == "train" else names[:val_size]
        self.root = root
        self.split = split

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        latent = np.load(self.root / "latent" / f"{name}.npy").astype(np.float32)
        gt = np.load(self.root / "gt" / f"{name}.npy").astype(np.float32)

        latent = torch.from_numpy(latent)
        # [0, 1] -> [-1, 1] agar sepadan dengan keluaran vae.decode(...).sample
        gt = torch.from_numpy(gt) * 2.0 - 1.0
        return latent, gt


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


# --------------------------------------------------------------------------- #
# Evaluasi ringkas saat pelatihan
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(vae, loader, criterion, scaling_factor, device, amp_dtype):
    vae.eval()
    tot = {"loss_total": 0.0, "psnr": 0.0, "spec": 0.0}
    n = 0
    for latent, gt in loader:
        latent, gt = latent.to(device), gt.to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            rec = vae.decode(latent / scaling_factor).sample
        rec = rec.float().clamp(-1, 1)
        _, parts = criterion(rec, gt)
        tot["loss_total"] += parts["loss_total"]
        tot["psnr"] += psnr(rec, gt)
        tot["spec"] += spectral_consistency(rec, gt)
        n += 1
    vae.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_dir", type=str, required=True, help="hasil cache_latents.py")
    ap.add_argument("--val_size", type=int, default=32)
    # model
    ap.add_argument("--sd_path", type=str, default="stabilityai/sd-turbo")
    ap.add_argument("--mode", type=str, default="partial", choices=["partial", "full"])
    ap.add_argument("--inner_dim", type=int, default=64, help="turunkan ke 32 bila VRAM kurang")
    # optimisasi
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", type=str, default="bf16", choices=["off", "fp16", "bf16"])
    # loss
    ap.add_argument("--pixel_type", type=str, default="l1", choices=["l1", "l2"])
    ap.add_argument("--w_pixel", type=float, default=1.0)
    ap.add_argument("--w_freq", type=float, default=0.1)
    ap.add_argument("--w_lpips", type=float, default=0.0)
    ap.add_argument("--freq_mode", type=str, default="full", choices=["full", "highpass"])
    # logging & checkpoint
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123456)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    amp_dtype = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # ----------------------------------------------------------------- model
    # Hanya VAE yang dimuat. Bobot dijaga float32: modul FGA mengandung LayerNorm
    # dan softmax attention yang rawan tidak stabil bila seluruhnya fp16.
    # Presisi campuran ditangani lewat autocast, bukan lewat bobot fp16.
    vae = AutoencoderKL.from_pretrained(
        args.sd_path, subfolder="vae", torch_dtype=torch.float32
    ).to(device)

    vae.requires_grad_(False)  # bekukan SELURUH backbone VAE lebih dulu
    trainable = inject_fga(vae, mode=args.mode, inner_dim=args.inner_dim)
    for p in trainable:
        p.requires_grad_(True)  # hanya modul FGA yang dibuka

    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in vae.parameters())
    print(f"[model] mode={args.mode} | parameter dilatih = {n_train:,} "
          f"({100 * n_train / n_total:.3f}% dari VAE)")

    # Gradient checkpointing pada decoder untuk menekan puncak memori.
    vae.decoder.gradient_checkpointing = True
    vae.train()

    scaling_factor = vae.config.scaling_factor
    print(f"[model] vae.config.scaling_factor = {scaling_factor}")

    # ------------------------------------------------------------------ data
    ds_tr = LatentHRDataset(args.data_dir, "train", args.val_size)
    ds_va = LatentHRDataset(args.data_dir, "val", args.val_size)
    print(f"[data] train={len(ds_tr)} | val={len(ds_va)}")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=2)
    stream = infinite(dl_tr)

    # ------------------------------------------------------------------ loss
    criterion = FGALoss(
        pixel_type=args.pixel_type,
        w_pixel=args.w_pixel,
        w_freq=args.w_freq,
        w_lpips=args.w_lpips,
        freq_mode=args.freq_mode,
    ).to(device)

    # ------------------------------------------------------------- optimizer
    # AdamW: weight decay ter-decouple, membantu regularisasi modul kecil pada
    # data fine-tuning yang terbatas. Konsisten pula dengan preseden InvSR yang
    # memakai keluarga Adam untuk melatih noise predictor-nya.
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16"))

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.iters - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))  # cosine decay

    # --------------------------------------------------------------- training
    history, best_psnr, t0 = [], -1e9, time.time()

    for step in range(args.iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        opt.zero_grad(set_to_none=True)
        acc = {}

        for _ in range(args.accum):
            latent, gt = next(stream)
            latent, gt = latent.to(device, non_blocking=True), gt.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                rec = vae.decode(latent / scaling_factor).sample

            # Loss dihitung di float32 (FFT tidak stabil di half precision)
            loss, parts = criterion(rec.float(), gt)
            scaler.scale(loss / args.accum).backward()

            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v / args.accum

        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        scaler.step(opt)
        scaler.update()

        if (step + 1) % args.log_every == 0:
            el = time.time() - t0
            msg = " | ".join(f"{k}={v:.5f}" for k, v in acc.items())
            print(f"[{step + 1}/{args.iters}] lr={lr_at(step):.2e} | {msg} | {el / 60:.1f} mnt")
            history.append({"step": step + 1, "lr": lr_at(step), **acc})

        if (step + 1) % args.val_every == 0:
            m = evaluate(vae, dl_va, criterion, scaling_factor, device, amp_dtype)
            print(f"    -> VAL loss={m['loss_total']:.5f} "
                  f"psnr={m['psnr']:.3f} dB spec={m['spec']:.4f}")
            history.append({"step": step + 1, "val": m})
            if m["psnr"] > best_psnr:
                best_psnr = m["psnr"]
                save_fga(vae, out_dir / f"fga_{args.mode}_best.pth", args, step + 1, m)
                print(f"    -> checkpoint terbaik disimpan (PSNR {best_psnr:.3f} dB)")

        if (step + 1) % args.save_every == 0:
            save_fga(vae, out_dir / f"fga_{args.mode}_last.pth", args, step + 1, None)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    save_fga(vae, out_dir / f"fga_{args.mode}_final.pth", args, args.iters, None)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[selesai] {(time.time() - t0) / 60:.1f} menit | PSNR val terbaik = {best_psnr:.3f} dB")


def save_fga(vae, path: Path, args, step: int, metrics) -> None:
    """Menyimpan HANYA bobot modul FGA (bukan seluruh VAE).

    Ukuran file hasilnya hanya sekitar 1-2 MB, sesuai rancangan pada Bab II.
    """
    state = {k: v.cpu() for k, v in vae.state_dict().items() if ".fga." in k}
    torch.save(
        {
            "state_dict": state,
            "mode": args.mode,
            "inner_dim": args.inner_dim,
            "step": step,
            "metrics": metrics,
        },
        path,
    )


if __name__ == "__main__":
    main()
