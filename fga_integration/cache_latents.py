"""
cache_latents.py — Tahap pra-pelatihan: menyimpan latent keluaran InvSR ke disk.

MENGAPA PERLU?
    Seluruh backbone InvSR (VAE Encoder, Noise Predictor, U-Net SD-Turbo) DIBEKUKAN
    pada penelitian ini. Artinya, untuk citra LR yang sama, latent hasil reverse
    diffusion TIDAK PERNAH berubah selama pelatihan modul FGA.

    Menjalankan backbone berulang kali di setiap iterasi adalah pemborosan murni.
    Dengan meng-cache latent sekali di awal, pelatihan FGA berubah menjadi tugas
    ringan "latent -> piksel HR" yang muat nyaman di GPU T4/P100 gratis.

    Biaya penyimpanan sangat kecil: latent 64x64x4 float16 = 32 KB per citra,
    sehingga 20.000 citra hanya ~640 MB.

CARA PAKAI
    python cache_latents.py \
        --cfg_path configs/sample-sd-turbo.yaml \
        --lr_dir  data/pairs/lr \
        --gt_dir  data/pairs/gt \
        --out_dir data/cache/steps1 \
        --num_steps 1

    Ulangi untuk --num_steps 2 / 3 / 5 bila ingin melatih atau mengevaluasi
    FGA pada rezim langkah sampling lain (lihat H2 pada proposal).

CATATAN VERIFIKASI
    Skrip ini memanggil pipeline dengan `output_type="latent"` agar pipeline
    berhenti SEBELUM tahap decoding VAE. Konvensi ini standar pada pipeline
    diffusers. Jalankan sekali dengan --limit 4 untuk memastikan bentuk tensor
    yang dikembalikan adalah (B, 4, H/8, W/8) sebelum memproses seluruh dataset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Modul milik repositori InvSR (jalankan skrip ini dari root repo InvSR)
from sampler_invsr import BaseSampler, _positive, _negative
from utils import util_image


def _save_atomic(path: Path, arr: np.ndarray) -> None:
    """Tulis .npy lewat berkas sementara lalu rename.

    `np.save` tidak atomik: proses yang terbunuh di tengah penulisan
    meninggalkan berkas terpotong yang header-nya sah tetapi datanya kurang,
    sehingga baru meledak jauh kemudian saat training — `ValueError: cannot
    reshape array of size N into shape (...)`. Rename pada filesystem yang sama
    bersifat atomik, jadi berkas hanya pernah terlihat utuh atau tidak ada.
    """
    tmp = path.with_suffix(".npy.tmp")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _complete(path: Path) -> bool:
    """True bila .npy ada DAN ukurannya sesuai header-nya.

    Berkas terpotong tetap punya header yang sah — hanya datanya yang kurang.
    Karena itu `exists()` saja tidak cukup untuk memutuskan boleh dilewati.
    """
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            shape, _, dtype = np.lib.format._read_array_header(f, version)
            header_end = f.tell()
        expected = header_end + int(np.prod(shape)) * dtype.itemsize
        return path.stat().st_size >= expected
    except Exception:
        return False


class LatentCacher(BaseSampler):
    """Menjalankan pipeline InvSR sampai tahap latent, lalu menyimpannya."""

    @torch.no_grad()
    def cache(
        self,
        lr_dir: str,
        gt_dir: str,
        out_dir: str,
        num_steps: int,
        scale: int = 4,
        limit: int | None = None,
    ) -> None:
        lr_dir, gt_dir, out_dir = Path(lr_dir), Path(gt_dir), Path(out_dir)
        (out_dir / "latent").mkdir(parents=True, exist_ok=True)
        (out_dir / "gt").mkdir(parents=True, exist_ok=True)

        lr_paths = sorted(
            p for p in lr_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if limit is not None:
            lr_paths = lr_paths[:limit]

        timesteps = self.configs.timesteps[:num_steps]
        print(f"[cache] {len(lr_paths)} citra | num_steps={num_steps} | timesteps={timesteps}")

        for i, lr_path in enumerate(lr_paths):
            gt_path = gt_dir / lr_path.name
            if not gt_path.exists():
                print(f"[skip] ground truth tidak ditemukan: {gt_path}")
                continue

            out_lat = out_dir / "latent" / f"{lr_path.stem}.npy"
            out_gt = out_dir / "gt" / f"{lr_path.stem}.npy"
            # Periksa KEDUANYA dan periksa keutuhannya, bukan sekadar exists().
            # GT ditulis setelah latent, jadi interupsi di antara keduanya
            # meninggalkan latent utuh dengan GT hilang/terpotong — dan versi
            # lama melewatinya selamanya karena hanya melihat latent.
            if _complete(out_lat) and _complete(out_gt):
                continue
            if out_lat.exists() or out_gt.exists():
                print(f"[cache] memperbaiki cache rusak: {lr_path.stem}", flush=True)

            im_lr = util_image.imread(str(lr_path), chn="rgb", dtype="float32")
            im_lr = util_image.img2tensor(im_lr).cuda()  # (1, 3, h, w), [0, 1]

            target_size = (im_lr.shape[-2] * scale, im_lr.shape[-1] * scale)

            latent = self.sd_pipe(
                image=im_lr.type(torch.float16),
                prompt=[_positive],
                negative_prompt=[_negative] if self.configs.cfg_scale > 1.0 else None,
                target_size=target_size,
                timesteps=timesteps,
                guidance_scale=self.configs.cfg_scale,
                output_type="latent",  # <-- berhenti sebelum VAE decode
            ).images

            if i == 0:
                print(f"[cache] bentuk latent = {tuple(latent.shape)}  dtype={latent.dtype}")
                expected = (1, 4, target_size[0] // 8, target_size[1] // 8)
                if tuple(latent.shape) != expected:
                    raise RuntimeError(
                        f"Bentuk latent tidak sesuai harapan {expected}. "
                        "Periksa apakah pipeline mendukung output_type='latent'."
                    )

            _save_atomic(out_lat, latent.squeeze(0).cpu().numpy().astype(np.float16))

            # Simpan ground truth HR sebagai float16 agar loader ringan
            im_gt = util_image.imread(str(gt_path), chn="rgb", dtype="float32")
            im_gt = util_image.img2tensor(im_gt).squeeze(0)  # (3, H, W), [0, 1]
            if im_gt.shape[-2:] != torch.Size(target_size):
                im_gt = F.interpolate(
                    im_gt[None], size=target_size, mode="bicubic", align_corners=False
                ).squeeze(0).clamp(0, 1)
            _save_atomic(out_gt, im_gt.numpy().astype(np.float16))

            if (i + 1) % 200 == 0:
                print(f"[cache] {i + 1}/{len(lr_paths)}")

        print(f"[cache] selesai -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", type=str, required=True, help="YAML konfigurasi InvSR")
    ap.add_argument("--lr_dir", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_steps", type=int, default=1, choices=[1, 2, 3, 5])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="uji coba dengan N citra saja")
    args = ap.parse_args()

    configs = OmegaConf.load(args.cfg_path)
    configs.timesteps = configs.get("timesteps", [250, 200, 150, 100, 50])

    cacher = LatentCacher(configs)
    cacher.cache(
        lr_dir=args.lr_dir,
        gt_dir=args.gt_dir,
        out_dir=args.out_dir,
        num_steps=args.num_steps,
        scale=args.scale,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
