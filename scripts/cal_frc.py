#!/usr/bin/env python
"""
cal_frc.py — Fourier Ring Correlation (FRC) dan FRC-AUC terhadap ground truth.

MENGAPA METRIK INI
    PSNR dan SSIM didominasi komponen frekuensi rendah: spektrum citra natural
    meluruh kira-kira 1/f, sehingga sebagian besar energinya ada di band rendah
    dan kesalahan pada detail halus nyaris tidak terlihat pada kedua metrik itu.

    FRC menghitung korelasi SECARA TERPISAH untuk tiap cincin frekuensi, lalu
    tiap cincin diberi bobot sama. Hasilnya: kurva yang menunjukkan sampai
    frekuensi berapa rekonstruksi masih setia pada ground truth — persis
    pertanyaan yang diajukan penelitian ini.

DEFINISI
    Untuk tiap cincin q pada bidang Fourier:

        FRC(q) = Re[ sum F1 * conj(F2) ] / sqrt( sum|F1|^2 * sum|F2|^2 )

    Nilainya pada [-1, 1]; 1 berarti kedua citra identik secara fase dan
    amplitudo pada frekuensi tersebut. FRC-AUC adalah rata-rata kurva itu atas
    seluruh cincin, dinormalisasi ke [0, 1] pada sumbu frekuensi — satu skalar
    yang merangkum kesetiaan di SELURUH pita, bukan hanya band rendah.

    Komponen DC (q=0) dilewati: ia satu piksel dan selalu berkorelasi sempurna.

CARA PAKAI
    python scripts/cal_frc.py --gt_dir gt --sr_dir out/full_v2.1_s1
    python scripts/cal_frc.py --gt_dir gt --sr_dir out/baseline,out/full_v2.1_s1

    Beberapa direktori dipisah koma akan dibandingkan pada DAFTAR NAMA YANG SAMA
    — syarat kebenaran, bukan kenyamanan: konten scene sangat memengaruhi
    statistik frekuensi, jadi membandingkan kondisi pada gambar berbeda bisa
    membalik kesimpulan.

TIDAK BUTUH GPU
    Murni numpy + PIL. Empat puluh citra 512x512 selesai dalam hitungan detik.
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import numpy as np
from PIL import Image


def _luma(path: str) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def frc_curve(a: np.ndarray, b: np.ndarray, eps: float = 1e-12):
    """Kurva FRC antara dua citra grayscale berukuran sama.

    Returns:
        freq: frekuensi ternormalisasi tiap cincin, 0 (DC) sampai 1 (Nyquist)
        frc : nilai korelasi tiap cincin
    """
    h, w = a.shape
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)

    # jarak radial tiap bin Fourier, dalam satuan piksel frekuensi
    fy = np.fft.fftfreq(h) * h
    fx = np.fft.fftfreq(w) * w
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    ring = r.astype(np.int64).ravel()
    qmax = min(h, w) // 2

    num = np.bincount(ring, weights=(A * np.conj(B)).real.ravel(), minlength=qmax + 1)
    d1 = np.bincount(ring, weights=(np.abs(A) ** 2).ravel(), minlength=qmax + 1)
    d2 = np.bincount(ring, weights=(np.abs(B) ** 2).ravel(), minlength=qmax + 1)

    q = np.arange(1, qmax + 1)          # lewati DC
    frc = num[1:qmax + 1] / (np.sqrt(d1[1:qmax + 1] * d2[1:qmax + 1]) + eps)
    return q / qmax, frc


def evaluate(gt_dir: str, sr_dir: str, names: list) -> dict:
    curves = []
    for n in names:
        gt, sr = _luma(os.path.join(gt_dir, n)), _luma(os.path.join(sr_dir, n))
        if gt.shape != sr.shape:        # samakan ukuran bila berbeda
            h = min(gt.shape[0], sr.shape[0])
            w = min(gt.shape[1], sr.shape[1])
            gt, sr = gt[:h, :w], sr[:h, :w]
        freq, c = frc_curve(sr, gt)
        curves.append(c)
    C = np.mean(np.stack(curves), axis=0)
    # AUC = rata-rata kurva; cincin berjarak seragam sehingga ini setara luas
    # di bawah kurva pada sumbu frekuensi ternormalisasi [0, 1].
    band = lambda lo, hi: float(C[(freq >= lo) & (freq < hi)].mean())
    return {
        "FRC-AUC": float(C.mean()),
        "rendah (0.0-0.33)": band(0.0, 1 / 3),
        "menengah (0.33-0.67)": band(1 / 3, 2 / 3),
        "tinggi (0.67-1.0)": band(2 / 3, 1.0001),
        "n": len(curves),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--sr_dir", required=True,
                    help="satu direktori, atau beberapa dipisah koma")
    ap.add_argument("--log_name", default="", help="salin keluaran ke berkas")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.sr_dir.split(",") if d.strip()]
    gt_names = {os.path.basename(p) for p in glob(os.path.join(args.gt_dir, "*.png"))}

    # nama yang ada di GT DAN di semua direktori SR
    names = sorted(gt_names.intersection(
        *[{os.path.basename(p) for p in glob(os.path.join(d, "*.png"))} for d in dirs]
    ))
    assert names, "tidak ada nama berkas yang sama di GT dan semua direktori SR"

    out = [f"[frc] {len(names)} citra, dibandingkan pada daftar nama yang sama\n",
           f"{'kondisi':28}{'FRC-AUC':>10}{'rendah':>10}{'menengah':>11}{'tinggi':>9}"]
    base = None
    for d in dirs:
        m = evaluate(args.gt_dir, d, names)
        if base is None:
            base = m["FRC-AUC"]
        delta = f"  ({100 * (m['FRC-AUC'] / base - 1):+.1f}%)" if len(dirs) > 1 else ""
        out.append(f"{os.path.basename(d.rstrip('/')):28}{m['FRC-AUC']:10.4f}"
                   f"{m['rendah (0.0-0.33)']:10.4f}{m['menengah (0.33-0.67)']:11.4f}"
                   f"{m['tinggi (0.67-1.0)']:9.4f}{delta}")
    if len(dirs) > 1:
        out.append("\nSelisih dihitung terhadap direktori PERTAMA pada --sr_dir.")
    out.append("\nFRC-AUC pada [-1, 1]; makin tinggi makin setia ke GT di seluruh pita.")
    out.append("Kolom 'tinggi' yang paling relevan untuk rekonstruksi detail.")

    text = "\n".join(out)
    print(text)
    if args.log_name:
        with open(args.log_name, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
