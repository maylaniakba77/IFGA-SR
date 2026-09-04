"""
make_pairs.py — Tahap persiapan data (Fase 2): membuat pasangan LR/GT di disk.

MENGAPA PERLU?
    InvSR asli TIDAK pernah menyimpan citra LR ke disk. Degradasi Real-ESRGAN
    dijalankan on-the-fly di GPU pada setiap iterasi (lihat trainer.py:prepare_data),
    dan RealESRGANDataset hanya mengembalikan 'gt' beserta tiga kernel acak.

    Sebaliknya, cache_latents.py membutuhkan pasangan STATIS di dua direktori
    dengan nama berkas yang sama. Skrip ini menjembatani keduanya: ia memakai
    kelas degradasi milik repositori ini (bukan implementasi ulang), lalu
    menuliskan hasilnya sebagai berkas PNG.

    Dengan begitu protokol degradasi tetap identik dengan InvSR, sehingga
    perbandingan terhadap baseline tetap sah.

CARA PAKAI
    python fga_integration/make_pairs.py \
        --hr_dir  data/source_hr \
        --out_dir data/pairs \
        --gt_size 256 \
        --draws   4

CATATAN PENTING
    --draws N menghasilkan N sampel degradasi INDEPENDEN per citra HR
    (0001_d0.png, 0001_d1.png, ...). Ini penting: caching latent membekukan
    satu degradasi per berkas, sedangkan InvSR melihat undian baru setiap epoch.
    Beberapa undian per citra memulihkan sebagian ragam tersebut.

    Keluaran selalu PNG (lossless). Menyimpan sebagai JPEG akan menambah
    artefak yang tidak terkendali di atas degradasi yang sudah disintesis.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from basicsr.data.realesrgan_dataset import RealESRGANDataset
from basicsr.utils import DiffJPEG

IM_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def build_dataset(hr_dir: Path, gt_size: int, random_crop: bool, cfg) -> RealESRGANDataset:
    """Membangun RealESRGANDataset yang menunjuk ke direktori HR milik pengguna.

    Parameter kernel diambil apa adanya dari configs/sd-turbo-sr-ldis.yaml agar
    identik dengan pelatihan InvSR.
    """
    p = cfg.data.train.params
    opt = {
        "data_source": {
            "source1": {
                "root_path": str(hr_dir.parent),
                "image_path": hr_dir.name,
                "moment_path": None,
                "text_path": None,
                "im_ext": "png",     # ditimpa di bawah; kita isi image_paths manual
                "length": None,
            }
        },
        "io_backend": {"type": "disk"},
        "max_token_length": p.max_token_length,
        # kernel degradasi pertama
        "blur_kernel_size": p.blur_kernel_size,
        "kernel_list": OmegaConf.to_object(p.kernel_list),
        "kernel_prob": OmegaConf.to_object(p.kernel_prob),
        "blur_sigma": OmegaConf.to_object(p.blur_sigma),
        "betag_range": OmegaConf.to_object(p.betag_range),
        "betap_range": OmegaConf.to_object(p.betap_range),
        "sinc_prob": p.sinc_prob,
        # kernel degradasi kedua
        "blur_kernel_size2": p.blur_kernel_size2,
        "kernel_list2": OmegaConf.to_object(p.kernel_list2),
        "kernel_prob2": OmegaConf.to_object(p.kernel_prob2),
        "blur_sigma2": OmegaConf.to_object(p.blur_sigma2),
        "betag_range2": OmegaConf.to_object(p.betag_range2),
        "betap_range2": OmegaConf.to_object(p.betap_range2),
        "sinc_prob2": p.sinc_prob2,
        "final_sinc_prob": p.final_sinc_prob,
        # pemotongan / augmentasi
        "gt_size": gt_size,
        "use_hflip": True,
        "use_rot": False,
        "random_crop": random_crop,
    }
    return RealESRGANDataset(OmegaConf.create(opt), mode="training")


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) float [0,1] RGB -> (H, W, 3) uint8."""
    a = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (a * 255.0).round().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_dir", type=str, required=True, help="direktori citra HR sumber")
    ap.add_argument("--out_dir", type=str, required=True, help="menerima lr/ dan gt/")
    ap.add_argument("--cfg_path", type=str, default="configs/sd-turbo-sr-ldis.yaml",
                    help="sumber parameter degradasi InvSR")
    ap.add_argument("--gt_size", type=int, default=256,
                    help="ukuran GT; LR = gt_size / sf. TIDAK dapat diubah setelah caching")
    ap.add_argument("--draws", type=int, default=1,
                    help="jumlah undian degradasi independen per citra HR")
    ap.add_argument("--random_crop", action="store_true", default=True)
    ap.add_argument("--center_crop", dest="random_crop", action="store_false",
                    help="pakai resize+center-crop, aman untuk citra kecil")
    ap.add_argument("--limit", type=int, default=None, help="uji coba dengan N citra saja")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123456)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hr_dir = Path(args.hr_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "lr").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.cfg_path)
    deg = cfg.degradation.copy()
    deg.sf = cfg.degradation.sf
    sf = int(deg.sf)
    if args.gt_size % sf != 0:
        raise ValueError(f"--gt_size {args.gt_size} harus habis dibagi sf={sf}")

    ds = build_dataset(hr_dir, args.gt_size, args.random_crop, cfg)

    # Daftar berkas diisi manual agar semua ekstensi didukung dan --limit sederhana.
    paths = sorted(p for p in hr_dir.iterdir() if p.suffix.lower() in IM_EXTS)
    if not paths:
        raise FileNotFoundError(f"Tidak ada citra di {hr_dir}")

    # random_crop membutuhkan citra minimal sebesar gt_size; sisanya dilewati.
    if args.random_crop:
        keep = []
        for p in paths:
            with Image.open(p) as im:
                if min(im.size) >= args.gt_size:
                    keep.append(p)
                else:
                    print(f"[skip] {p.name} lebih kecil dari gt_size ({im.size})")
        paths = keep
        if not paths:
            raise RuntimeError(
                f"Semua citra lebih kecil dari --gt_size {args.gt_size}. "
                "Gunakan --center_crop atau turunkan --gt_size."
            )

    if args.limit is not None:
        paths = paths[: args.limit]

    ds.image_paths = [str(p) for p in paths]
    ds.text_paths = [None] * len(paths)
    ds.moment_paths = [None] * len(paths)

    # degrade_fun membuat DiffJPEG di CPU saat pertama dipakai; siapkan di device
    # yang benar lebih dulu agar tidak terjadi ketidakcocokan perangkat.
    ds.jpeger = DiffJPEG(differentiable=False).to(args.device)

    print(f"[pairs] {len(paths)} citra HR x {args.draws} undian = "
          f"{len(paths) * args.draws} pasangan | GT {args.gt_size} -> LR {args.gt_size // sf}")

    written = 0
    for i in range(len(paths)):
        stem = Path(ds.image_paths[i]).stem
        for d in range(args.draws):
            name = f"{stem}.png" if args.draws == 1 else f"{stem}_d{d}.png"
            lr_out, gt_out = out_dir / "lr" / name, out_dir / "gt" / name
            if lr_out.exists() and gt_out.exists():
                continue  # aman untuk dilanjutkan setelah interupsi

            item = ds[i]  # crop + augmentasi + kernel acak baru setiap pemanggilan
            with torch.no_grad():
                res = ds.degrade_fun(
                    deg,
                    item["gt"][None].to(args.device),
                    item["kernel1"][None].to(args.device),
                    item["kernel2"][None].to(args.device),
                    item["sinc_kernel"][None].to(args.device),
                )
            Image.fromarray(to_uint8(res["gt"])).save(gt_out)
            Image.fromarray(to_uint8(res["lq"])).save(lr_out)
            written += 1

        if (i + 1) % 100 == 0:
            print(f"[pairs] {i + 1}/{len(paths)} citra")

    print(f"[pairs] selesai: {written} pasangan baru -> {out_dir}")
    print(f"[pairs] lanjutkan ke cache_latents.py dengan "
          f"--lr_dir {out_dir}/lr --gt_dir {out_dir}/gt")


if __name__ == "__main__":
    main()
