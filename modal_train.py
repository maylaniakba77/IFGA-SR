"""
modal_train.py — menjalankan seluruh pipeline FGA di Modal.

MENGAPA MODAL, BUKAN COLAB
    train_fga.py TIDAK punya resume: loop-nya `for step in range(args.iters)` dan
    save_fga hanya menyimpan bobot `.fga.` — tanpa state optimizer, scaler, atau
    scheduler. Di Colab, disconnect di jam ke-9 menghanguskan seluruh run.
    Modal memberi timeout eksplisit (di sini 24 jam) tanpa idle-kick.

    Kedua: kode ini lebih suka bf16 (FGA berisi softmax attention + LayerNorm
    yang tidak stabil di fp16), tetapi Colab free memberi T4 yang tidak punya
    bf16 native. Di sini GPU dipilih eksplisit — L4 (Ada) punya bf16.

STRUKTUR
    Setiap fase = satu Modal function yang memanggil skrip CLI yang sudah ada
    lewat subprocess. Sengaja begitu: perintahnya identik dengan TRAINING.md,
    jadi tidak ada jalur kode kedua yang bisa menyimpang dari dokumentasi.

CARA PAKAI
    Lihat MODAL.md. Ringkasnya:
        modal run modal_train.py::prepare
        modal run modal_train.py::make_pairs
        modal run modal_train.py::cache
        modal run modal_train.py::train --mode partial
"""

import os

import modal

REPO = "/repo"       # repositori ini, dimount dari mesin lokal
VOL = "/vol"         # Volume persisten

# GPU dibaca dari environment SAAT IMPORT, karena argumen `gpu=` pada
# @app.function dievaluasi saat dekorasi — bukan saat pemanggilan. Jadi:
#     IFGA_GPU=A100-40GB modal run modal_train.py::train --mode partial
#
# Naik tier belum tentu menolong. Pada --batch 1 dengan crop 256, satu langkah
# hanya men-decode latent 32x32x4 menjadi 256x256x3, dan FGA dengan window_size=1
# menghasilkan h*w window attention yang masing-masing cuma 4 query x 25 key.
# Beban itu dibatasi overhead kernel dan bandwidth, bukan FLOPs, sehingga GPU
# besar akan menganggur. Pengungkit sebenarnya adalah --batch (lihat MODAL.md §9).
GPU_TRAIN = os.environ.get("IFGA_GPU", "L4")   # Ada: bf16 native, 24 GB
GPU_CACHE = os.environ.get("IFGA_GPU_CACHE", "L4")
GPU_CHEAP = "T4"     # sintesis degradasi: murah, tidak butuh bf16

TIMESTEPS = {1: [200], 2: [200, 100], 3: [200, 100, 50],
             4: [200, 150, 100, 50], 5: [250, 200, 150, 100, 50]}

# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #
# CATATAN DEPENDENSI — jangan ikuti daftar "skip" di notebook Colab.
#
# Notebook menyebut `albumentations` sebagai "demo/eval only" dan tidak
# memasangnya. Itu hanya berhasil karena Colab SUDAH memuatnya lebih dulu.
# Kenyataannya `basicsr/data/realesrgan_dataset.py` mengimpornya di level modul,
# dan sampler_invsr.py -> datapipe.datasets -> realesrgan_dataset, sehingga
# albumentations wajib ada bahkan untuk caching dan inferensi.
#
# `pip install -e .` juga dilewati di sini: setup.py repo ini adalah milik
# diffusers, dan satu-satunya gunanya di notebook adalah membuat basicsr
# ditemukan — yang sudah dikerjakan PYTHONPATH di bawah. Melewatinya berarti
# perubahan kode lokal langsung terpakai tanpa rebuild image.
#
# STRUKTUR IMAGE — mengapa `_base` dipisah dari `image`
#
# Modal menolak build step apa pun SETELAH `add_local_*`:
#     "An image tried to run a build step after using image.add_local_* ..."
# Jadi `metrics_image = image.pip_install(...)` tidak sah bila `image` sudah
# memuat berkas lokal. Solusinya: simpan `_base` tanpa berkas lokal, turunkan
# setiap varian dari situ, lalu tempelkan `_with_repo()` paling akhir.
_base = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")  # cv2 butuh libGL
    .pip_install("torch==2.4.0", "torchvision==0.19.0")
    .pip_install(
        # jalur data / degradasi (basicsr, datapipe, utils)
        "albumentations==1.4.18", "opencv-python==4.10.0.84",
        "scipy", "scikit-image", "lmdb", "requests", "tqdm", "pyyaml", "einops",
        # konfigurasi & logging
        "omegaconf", "loguru", "python-box",
        # dependensi diffusers yang di-vendor di src/
        "transformers==4.37.2", "peft==0.7.1", "huggingface_hub",
        "safetensors", "accelerate", "Pillow", "packaging", "regex",
        # --w_lpips > 0 dan --select_by lpips
        "lpips",
    )
    .env({
        # src/ HARUS mendahului agar diffusers yang di-vendor tidak ter-shadow
        "PYTHONPATH": f"{REPO}:{REPO}/src",
        # tanpa ini sd-turbo di-download ulang setiap cold start
        "HF_HOME": f"{VOL}/models",
        "HF_HUB_CACHE": f"{VOL}/models/hub",
    })
)


def _with_repo(img: modal.Image) -> modal.Image:
    """Tempelkan repositori lokal. HARUS jadi lapisan terakhir.

    copy=False: berkas dimount saat container start, bukan dibakar ke image,
    sehingga mengedit losses.py tidak memicu rebuild.
    """
    return img.add_local_dir(
        ".", REPO,
        ignore=["**/.git", "**/__pycache__", "**/*.pyc", "assets/**",
                "testdata/**", "notebooks/**", "data/**", "experiments/**"],
    )


image = _with_repo(_base)

# pyiqa berat dan hanya dipakai scripts/cal_metrics_ref.py, jadi ia diturunkan
# dari _base (bukan dari `image`) supaya pip_install-nya tetap sebelum add_local_dir
metrics_image = _with_repo(_base.pip_install("pyiqa==0.1.12"))

app = modal.App("ifga-sr")
vol = modal.Volume.from_name("ifga-sr", create_if_missing=True)


# --------------------------------------------------------------------------- #
# Helper (dijalankan DI DALAM container)
# --------------------------------------------------------------------------- #
def _sh(*cmd: str) -> None:
    """Jalankan skrip repo dari /repo, gagalkan function bila exit code != 0."""
    import subprocess
    print("+", " ".join(cmd), flush=True)
    subprocess.run(list(cmd), cwd=REPO, check=True)


def _guard_empty(path: str, pattern: str, overwrite: bool, hint: str) -> None:
    """Tolak menulis ke direktori yang sudah berisi hasil.

    Perlu karena tidak ada satu pun skrip di repo ini yang memperingatkan:
    `inference_invsr.py` memakai mkdir(delete=False) lalu menulis PNG dengan
    nama berkas input yang sama, dan `save_fga` menimpa .pth apa adanya. Dua
    kali jalan dengan nama sama = hasil lama hilang tanpa jejak.
    """
    from pathlib import Path as _P
    d = _P(path)
    ada = sorted(d.glob(pattern)) if d.exists() else []
    if ada and not overwrite:
        raise RuntimeError(
            f"{path} sudah berisi {len(ada)} berkas {pattern} "
            f"(mis. {ada[0].name}).\n"
            f"Menjalankan ini akan menimpanya. Pilih salah satu:\n"
            f"  - {hint}\n"
            f"  - tambahkan --overwrite bila memang ingin menimpa"
        )


def _write_config(num_steps: int) -> str:
    """Setara Phase 3 pada notebook Colab.

    configs/sample-sd-turbo.yaml tidak bisa dipakai apa adanya:
      1. model_start.ckpt_path = ~   -> sampler_invsr.py assert gagal
      2. cache_dir menunjuk path cluster penulis asli
      3. timesteps default caching [250,...] BEDA dari jadwal inferensi [200]
         -- melatih di satu jadwal lalu mengevaluasi di jadwal lain adalah
         train/test mismatch di dalam bagian pipeline yang dibekukan.
    """
    import os
    import shutil
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    os.makedirs(f"{VOL}/weights", exist_ok=True)
    os.makedirs(f"{VOL}/configs", exist_ok=True)

    ckpt = f"{VOL}/weights/noise_predictor_sd_turbo_v5.pth"
    if not os.path.exists(ckpt):
        shutil.copy(hf_hub_download("OAOA/InvSR", "noise_predictor_sd_turbo_v5.pth"), ckpt)
        print(f"noise predictor -> {ckpt}", flush=True)

    cfg = OmegaConf.load(f"{REPO}/configs/sample-sd-turbo.yaml")
    cfg.model_start.ckpt_path = ckpt
    cfg.sd_pipe.params.cache_dir = f"{VOL}/models"
    cfg.timesteps = TIMESTEPS[num_steps]

    out = f"{VOL}/configs/modal-sd-turbo.yaml"
    OmegaConf.save(cfg, out)
    print(f"config -> {out} | timesteps={cfg.timesteps}", flush=True)
    vol.commit()
    return out


# --------------------------------------------------------------------------- #
# Fase 1 — citra HR sumber
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={VOL: vol}, timeout=3600)
def prepare(dataset: str = "valid"):
    """Unduh DIV2K HR ke Volume. 'valid'=100 citra/449MB, 'train'=800/3.5GB.

    Unduhan memakai `requests`, BUKAN wget: debian_slim tidak memuat wget, dan
    menambahkannya ke apt_install akan membatalkan cache lapisan pip di atasnya
    sehingga torch (~2,5 GB) ikut ter-build ulang. requests sudah ada di image.

    Ekstraksi menulis PNG LANGSUNG ke Volume. Jangan tergoda mengekstrak ke /tmp
    lalu os.rename ke Volume: keduanya filesystem berbeda, jadi rename gagal
    dengan `OSError: [Errno 18] Invalid cross-device link`.
    """
    import os
    import zipfile

    import requests

    src = f"{VOL}/source_hr"
    os.makedirs(src, exist_ok=True)
    ada = [f for f in os.listdir(src) if f.lower().endswith(".png")]
    if ada:
        print(f"{len(ada)} citra sudah ada di {src} — dilewati")
        return

    url = f"https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_{dataset}_HR.zip"
    zp = "/tmp/div2k.zip"
    print(f"mengunduh {url}", flush=True)
    with requests.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = next_log = 0
        with open(zp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done >= next_log:
                    print(f"  {done / 1e9:.2f} GB"
                          f"{f' / {total / 1e9:.2f} GB' if total else ''}", flush=True)
                    next_log = done + (1 << 27)  # tiap ~134 MB
    # content-length yang tidak cocok = unduhan terpotong; zipfile akan gagal
    # dengan pesan yang membingungkan, jadi tangkap di sini.
    assert not total or done == total, f"unduhan terpotong: {done}/{total} byte"
    print(f"selesai: {done / 1e9:.2f} GB", flush=True)

    n = 0
    with zipfile.ZipFile(zp) as z:
        for member in z.namelist():
            if not member.lower().endswith(".png"):
                continue
            with z.open(member) as fsrc, open(f"{src}/{os.path.basename(member)}", "wb") as fdst:
                fdst.write(fsrc.read())
            n += 1
    assert n, "zip terekstrak tetapi tidak ada PNG di dalamnya"
    os.remove(zp)
    vol.commit()
    print(f"{n} citra HR -> {src}")


# --------------------------------------------------------------------------- #
# Fase 2 — pasangan LR/GT
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU_CHEAP, volumes={VOL: vol}, timeout=6 * 3600)
def make_pairs(gt_size: int = 512, draws: int = 4, limit: int = 0):
    """Sintesis LR memakai degradasi Real-ESRGAN milik repo ini.

    gt_size 512 (bukan 256 seperti notebook) supaya --crop 256 saat training
    benar-benar berfungsi: crop dilewati bila ukurannya >= citra.
    """
    vol.reload()
    cmd = ["python", "fga_integration/make_pairs.py",
           "--hr_dir", f"{VOL}/source_hr", "--out_dir", f"{VOL}/pairs",
           "--gt_size", str(gt_size), "--draws", str(draws)]
    if limit:
        cmd += ["--limit", str(limit)]
    _sh(*cmd)
    vol.commit()


# --------------------------------------------------------------------------- #
# Fase 3-4 — cache latent
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU_CACHE, volumes={VOL: vol}, timeout=12 * 3600)
def cache(num_steps: int = 1, limit: int = 0):
    """Jalankan backbone InvSR yang dibekukan sekali, simpan latent + GT."""
    vol.reload()
    cfg = _write_config(num_steps)
    cmd = ["python", "fga_integration/cache_latents.py",
           "--cfg_path", cfg,
           "--lr_dir", f"{VOL}/pairs/lr", "--gt_dir", f"{VOL}/pairs/gt",
           "--out_dir", f"{VOL}/cache/steps{num_steps}",
           "--num_steps", str(num_steps)]
    if limit:
        cmd += ["--limit", str(limit)]
    _sh(*cmd)
    vol.commit()


@app.function(image=image, gpu=GPU_CACHE, volumes={VOL: vol}, timeout=1800)
def gate_roundtrip(num_steps: int = 1):
    """GATE 1b — konvensi scaling latent.

    cache_latents.py menyimpan latent MENTAH; train_fga.py membaginya dengan
    vae.config.scaling_factor sebelum decode. Kalau konvensi ini tidak cocok,
    FGA akan dilatih untuk mengompensasi bug, bukan untuk menambah detail.
    Hasilnya ditulis sebagai PNG ke Volume — ambil dengan `modal volume get`.
    """
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL

    vol.reload()
    lat_dir = Path(f"{VOL}/cache/steps{num_steps}/latent")
    paths = sorted(lat_dir.glob("*.npy"))
    assert paths, f"tidak ada latent di {lat_dir} -- jalankan `cache` lebih dulu"

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-turbo", subfolder="vae", torch_dtype=torch.float32
    ).cuda().eval()
    lat = torch.from_numpy(np.load(paths[0]).astype("float32"))[None].cuda()
    with torch.no_grad():
        rec = vae.decode(lat / vae.config.scaling_factor).sample

    img = ((rec.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
    out = f"{VOL}/gates/roundtrip.png"
    Path(f"{VOL}/gates").mkdir(exist_ok=True)
    Image.fromarray((img * 255).round().astype("uint8")).save(out)
    vol.commit()
    print(f"latent {tuple(lat.shape)} -> {tuple(rec.shape)} | "
          f"scaling_factor={vae.config.scaling_factor}")
    print(f"LULUS bila gambar ini terlihat wajar (bukan noise abu-abu): {out}")


# --------------------------------------------------------------------------- #
# Fase 6-7 — training
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU_TRAIN, volumes={VOL: vol}, timeout=24 * 3600)
def train(
    mode: str = "partial",
    iters: int = 10000,
    accum: int = 4,
    batch: int = 1,
    crop: int = 256,
    num_steps: int = 1,
    freq_mode: str = "magnitude",
    freq_cutoff: float = 0.25,
    pixel_type: str = "l1",
    w_pixel: float = 1.0,
    w_freq: float = 1.0,
    w_lpips: float = 0.5,
    lpips_net: str = "alex",
    select_by: str = "lpips",
    inner_dim: int = 64,
    lr: float = 1e-4,
    amp: str = "bf16",
    val_size: int = 32,
    seed: int = 123456,
    log_every: int = 50,
    val_every: int = 500,
    tag: str = "mag",
    overwrite: bool = False,
):
    """Latih modul FGA. Default di sini adalah konfigurasi yang MENDORONG ketajaman.

    freq_mode='magnitude' + select_by='lpips' disengaja. Default skripnya
    ('full' + 'psnr') menghasilkan model yang lebih HALUS dari baseline: loss
    FFT kompleks sensitif fase sehingga meredam detail yang tidak sejajar
    piksel, dan seleksi PSNR memilih varian paling halus di antara semua step.
    Untuk mereproduksi kondisi pembanding itu, panggil dengan
    freq_mode='full' w_freq=0.1 w_lpips=0.0 select_by='psnr' tag='base'.
    """
    vol.reload()
    _guard_empty(f"{VOL}/experiments/fga_{mode}_{tag}", "*.pth", overwrite,
                 "pakai --tag <nama lain>")

    _sh("python", "fga_integration/train_fga.py",
        "--data_dir", f"{VOL}/cache/steps{num_steps}",
        "--mode", mode,
        "--iters", str(iters), "--batch", str(batch), "--accum", str(accum),
        "--crop", str(crop), "--inner_dim", str(inner_dim),
        "--lr", str(lr), "--amp", amp, "--seed", str(seed),
        "--val_size", str(val_size),
        "--pixel_type", pixel_type,
        "--w_pixel", str(w_pixel), "--w_freq", str(w_freq),
        "--freq_mode", freq_mode, "--freq_cutoff", str(freq_cutoff),
        "--w_lpips", str(w_lpips), "--lpips_net", lpips_net,
        "--select_by", select_by,
        "--val_every", str(val_every), "--save_every", str(val_every),
        "--log_every", str(log_every),
        "--out_dir", f"{VOL}/experiments/fga_{mode}_{tag}")
    vol.commit()


@app.function(image=image, gpu=GPU_TRAIN, volumes={VOL: vol}, timeout=1800)
def gate2(num_steps: int = 1, inner_dim: int = 64, n: int = 4):
    """GATE 2 — verifikasi keempat kriteria lulus secara langsung, tanpa training.

    Kriteria 3 ("loss step 0 setara baseline") tidak bisa dibaca dari log
    training: log pertama baru muncul di step ke-`log_every`, dan tidak ada
    angka baseline untuk dibandingkan. Di sini baseline dihitung eksplisit
    dengan mode="none", lalu dibandingkan terhadap keluaran partial/full yang
    BELUM dilatih.

    Karena `unembed` di-zero-init, delta = conv(bobot 0, bias 0) = tepat 0.0,
    sehingga keluaran harus IDENTIK BIT dengan baseline — bukan sekadar "mirip".
    Selisih bukan-nol berarti zero-init tidak berlaku dan cabang residual sudah
    merusak prior pretrained sejak iterasi pertama.

    Dijalankan di float32 tanpa autocast supaya perbandingannya eksak.
    """
    import sys
    from pathlib import Path

    import numpy as np
    import torch

    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from diffusers import AutoencoderKL
    from fga_integration.patch_decoder import inject_fga

    vol.reload()
    root = Path(f"{VOL}/cache/steps{num_steps}")
    names = sorted(p.stem for p in (root / "latent").glob("*.npy"))[:n]
    assert names, f"cache kosong di {root} — jalankan `cache` lebih dulu"
    print(f"[gate2] menguji {len(names)} sampel dari {root}\n")

    def load(name):
        lat = torch.from_numpy(
            np.load(root / "latent" / f"{name}.npy").astype("float32"))[None].cuda()
        gt = torch.from_numpy(
            np.load(root / "gt" / f"{name}.npy").astype("float32"))[None].cuda() * 2 - 1
        return lat, gt

    def build(mode):
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-turbo", subfolder="vae", torch_dtype=torch.float32
        ).cuda().eval()
        vae.requires_grad_(False)
        # VAE baru untuk tiap mode: inject_fga memodifikasi decoder di tempat
        trainable = inject_fga(vae, mode=mode, inner_dim=inner_dim)
        return vae, sum(p.numel() for p in trainable), sum(p.numel() for p in vae.parameters())

    @torch.no_grad()
    def decode(vae, lat):
        return vae.decode(lat / vae.config.scaling_factor).sample.float()

    # ---------------------------------------------------------------- baseline
    base, _, n_total = build("none")
    ref, l1_base = {}, {}
    for nm in names:
        lat, gt = load(nm)
        ref[nm] = decode(base, lat)
        l1_base[nm] = float((ref[nm].clamp(-1, 1) - gt).abs().mean())
    del base
    torch.cuda.empty_cache()
    mean_base = sum(l1_base.values()) / len(l1_base)
    print(f"[baseline] mode=none | L1 rekonstruksi = {mean_base:.6f} "
          f"| parameter VAE = {n_total:,}\n")

    # ------------------------------------------------------------- dua varian
    hasil = {}
    for mode in ("partial", "full"):
        vae, n_tr, _ = build(mode)
        worst_diff, worst_l1, finite = 0.0, 0.0, True
        for nm in names:
            lat, gt = load(nm)
            rec = decode(vae, lat)
            finite &= bool(torch.isfinite(rec).all())
            worst_diff = max(worst_diff, float((rec - ref[nm]).abs().max()))
            worst_l1 = max(worst_l1,
                           abs(float((rec.clamp(-1, 1) - gt).abs().mean()) - l1_base[nm]))
        hasil[mode] = {"params": n_tr, "pct": 100 * n_tr / n_total,
                       "max_diff": worst_diff, "d_l1": worst_l1, "finite": finite}
        del vae
        torch.cuda.empty_cache()
        print(f"[{mode}] parameter dilatih = {n_tr:,} ({hasil[mode]['pct']:.3f}% dari VAE) "
              f"| max|rec - baseline| = {worst_diff:.3e} "
              f"| selisih L1 = {worst_l1:.3e} | finit = {finite}")

    # -------------------------------------------------------------- keputusan
    p_, f_ = hasil["partial"], hasil["full"]
    cek = [
        ("1. parameter dilatih bukan nol",
         p_["params"] > 0 and f_["params"] > 0,
         f"partial={p_['params']:,} full={f_['params']:,}"),
        ("2. parameter full > partial",
         f_["params"] > p_["params"],
         f"{f_['params']:,} > {p_['params']:,}" if f_["params"] > p_["params"]
         else "SAMA — pemilihan mode rusak, ablasi H3 tidak bermakna"),
        ("3. zero-init: keluaran identik baseline",
         p_["max_diff"] == 0.0 and f_["max_diff"] == 0.0,
         f"max|selisih| partial={p_['max_diff']:.3e} full={f_['max_diff']:.3e} "
         f"(harus tepat 0.0)"),
        ("4. loss finit",
         p_["finite"] and f_["finite"],
         "tidak ada nan/inf" if p_["finite"] and f_["finite"] else "ADA nan/inf"),
    ]
    print("\n" + "=" * 72)
    for label, ok, detail in cek:
        print(f"{'LULUS' if ok else 'GAGAL':6} | {label:42} | {detail}")
    print("=" * 72)
    print("\nCatat kedua jumlah parameter — keduanya masuk tabel hasil tesis.")
    assert all(ok for _, ok, _ in cek), "GATE 2 gagal — jangan lanjut ke training panjang"


# --------------------------------------------------------------------------- #
# Fase 8-9 — inferensi & metrik
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU_TRAIN, volumes={VOL: vol}, timeout=6 * 3600)
def infer(fga_mode: str = "none", tag: str = "mag", num_steps: int = 1,
          color_fix: str = "", split_from: int = 1, out_name: str = "",
          overwrite: bool = False):
    """Jalankan inferensi pada split validasi (yang tidak pernah dilihat FGA).

    `num_steps`  jadwal sampling yang DIPAKAI saat inferensi.
    `split_from` cache yang MENDEFINISIKAN daftar gambar validasi.

    Keduanya dipisah karena split val ditentukan oleh cache tempat model
    dilatih (biasanya steps1). Kalau `num_steps` juga dipakai untuk mencari
    daftar gambar, mengevaluasi di 5 step akan mencari `cache/steps5` yang
    mungkin tidak pernah dibuat — dan lebih buruk lagi, mengevaluasi rezim
    berbeda pada daftar gambar berbeda sehingga angkanya tidak sebanding.

    Nama direktori keluaran menyertakan jumlah step (`baseline_s5`), supaya
    hasil rezim berbeda tidak saling menimpa.

    PERINGATAN TRAIN/TEST MISMATCH
        Checkpoint FGA dilatih pada latent dari SATU jadwal sampling. Menjalankan
        `--num-steps 5` pada checkpoint yang dilatih di steps1 berarti modul itu
        melihat distribusi latent yang belum pernah dilatihkan. Itu eksperimen
        yang sah (menguji ketahanan lintas rezim, H2), tetapi HARUS dilaporkan
        sebagai itu — bukan sebagai "FGA di 5 step". Untuk yang terakhir, cache
        dan latih ulang di --num-steps 5.
    """
    import shutil
    from pathlib import Path

    vol.reload()
    cfg = _write_config(num_steps)

    # Rekonstruksi split val LatentHRDataset: val_size nama pertama (tersortir)
    split_dir = Path(f"{VOL}/cache/steps{split_from}/latent")
    names = sorted(p.stem for p in split_dir.glob("*.npy"))
    assert names, (f"tidak ada cache di {split_dir} — pakai --split-from N yang "
                   f"menunjuk cache tempat model dilatih")
    if num_steps != split_from:
        print(f"[infer] PERINGATAN: sampling {num_steps} step, tetapi split val "
              f"berasal dari cache steps{split_from}. Checkpoint FGA dilatih di "
              f"steps{split_from} — laporkan ini sebagai uji lintas rezim.", flush=True)
    val_names = names[:min(32, max(1, len(names) // 10))]
    eval_lr = Path("/tmp/eval/lr")
    eval_lr.mkdir(parents=True, exist_ok=True)
    for n in val_names:
        shutil.copy(f"{VOL}/pairs/lr/{n}.png", eval_lr / f"{n}.png")

    # color_fix masuk ke nama: tanpa ini, run polos dan run wavelet pada mode dan
    # rezim step yang sama akan bertabrakan di direktori yang sama.
    suffix = f"_s{num_steps}" + (f"_{color_fix}" if color_fix else "")
    name = out_name or (f"baseline{suffix}" if fga_mode == "none"
                        else f"{fga_mode}_{tag}{suffix}")
    # --sd_path dan --started_ckpt_path WAJIB. inference_invsr.py MENIMPA
    # cache_dir dan ckpt_path dari YAML dengan default './weights' (relatif ke
    # /repo, yang read-only di container), sehingga tanpa keduanya ia mencoba
    # mengunduh ulang sd-turbo ~2,5 GB setiap run -- atau langsung gagal menulis.
    _guard_empty(f"{VOL}/out/{name}", "*.png", overwrite,
                 "pakai --out-name <nama lain>, atau --tag yang berbeda")

    cmd = ["python", "inference_invsr.py",
           "-i", str(eval_lr), "-o", f"{VOL}/out/{name}",
           "--cfg_path", cfg, "-n", str(num_steps),
           "--sd_path", f"{VOL}/models",
           "--started_ckpt_path", f"{VOL}/weights/noise_predictor_sd_turbo_v5.pth",
           "--fga_mode", fga_mode]
    if fga_mode != "none":
        cmd += ["--fga_ckpt",
                f"{VOL}/experiments/fga_{fga_mode}_{tag}/fga_{fga_mode}_best.pth"]
    if color_fix:
        cmd += ["--color_fix", color_fix]
    _sh(*cmd)

    # GATE 3 — GT juga disalin supaya `metrics` punya referensi
    gt_dir = Path(f"{VOL}/eval/gt")
    gt_dir.mkdir(parents=True, exist_ok=True)
    for n in val_names:
        shutil.copy(f"{VOL}/pairs/gt/{n}.png", gt_dir / f"{n}.png")
    vol.commit()


@app.function(image=metrics_image, gpu=GPU_TRAIN, volumes={VOL: vol}, timeout=3600)
def metrics(name: str = "baseline", overwrite: bool = False):
    """PSNR/SSIM/LPIPS terhadap GT yang ditahan."""
    vol.reload()
    _guard_empty(f"{VOL}/out", f"metrics_{name}.log", overwrite,
                 "pakai --name yang berbeda")
    _sh("python", "scripts/cal_metrics_ref.py",
        "--gt_dir", f"{VOL}/eval/gt",
        "--sr_dir", f"{VOL}/out/{name}",
        "--log_name", f"{VOL}/out/metrics_{name}.log")
    vol.commit()


@app.function(image=image, volumes={VOL: vol}, timeout=600)
def gate_diff(tags: str = "partial_mag,full_mag"):
    """GATE 3 — keluaran varian HARUS berbeda dari baseline.

    Nilai 0.0 berarti bobot FGA tidak sampai ke model, dan seluruh metrik
    hilirnya hanya duplikat baseline -- terlihat seperti "FGA tidak berpengaruh"
    padahal itu bug wiring.
    """
    import os
    import numpy as np
    from PIL import Image

    vol.reload()
    base_dir = f"{VOL}/out/baseline"
    base = sorted(os.listdir(base_dir))
    assert base, "baseline belum dijalankan"

    def load(d, n):
        return np.asarray(Image.open(os.path.join(d, n))).astype(np.float64)

    for tag in tags.split(","):
        d = f"{VOL}/out/{tag}"
        if not os.path.isdir(d):
            print(f"{tag:16} | BELUM DIJALANKAN")
            continue
        diffs = [np.abs(load(d, n) - load(base_dir, n)).mean()
                 for n in base if os.path.exists(os.path.join(d, n))]
        m = float(np.mean(diffs))
        print(f"{tag:16} | mean abs diff vs baseline = {m:.4f} (skala 0-255) | "
              f"{'GAGAL -- checkpoint tidak termuat' if m == 0.0 else 'OK'}")
