# Menjalankan Pelatihan FGA di Modal

Runbook ini adalah padanan [`notebooks/FGA_Training_Colab.ipynb`](notebooks/FGA_Training_Colab.ipynb)
untuk [Modal](https://modal.com). Penomoran fasenya mengikuti
[`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md), sehingga kegagalan di sini bisa
dipetakan ke fase bernama di dokumen itu.

Seluruh perintah dijalankan dari **root repositori** di mesin lokalmu. Yang
berjalan di GPU adalah [`modal_train.py`](modal_train.py).

---

## Contents

- [1. Mengapa Modal untuk proyek ini](#1-mengapa-modal-untuk-proyek-ini)
- [2. Biaya](#2-biaya)
- [3. Persiapan sekali jalan](#3-persiapan-sekali-jalan)
- [4. Peta fase](#4-peta-fase)
- [5. Fase 1 — Citra HR sumber](#5-fase-1--citra-hr-sumber)
- [6. Fase 2 — Pasangan LR/GT](#6-fase-2--pasangan-lrgt)
- [7. Fase 3-4 — Cache latent + GATE 1b](#7-fase-3-4--cache-latent--gate-1b)
- [8. Fase 5 — GATE 2: smoke test](#8-fase-5--gate-2-smoke-test)
- [9. Fase 6-7 — Latih `partial` dan `full`](#9-fase-6-7--latih-partial-dan-full)
- [10. Fase 8-9 — Inferensi, GATE 3, metrik](#10-fase-8-9--inferensi-gate-3-metrik)
- [11. Mengambil hasil](#11-mengambil-hasil)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Checklist](#13-checklist)

---

## 1. Mengapa Modal untuk proyek ini

Tiga alasan yang spesifik ke kode di repo ini, bukan preferensi umum.

**`train_fga.py` tidak punya resume.** Loop-nya `for step in range(args.iters)`
dan `save_fga` hanya menulis bobot `.fga.` — tanpa state optimizer, `GradScaler`,
atau LR scheduler. Disconnect di jam ke-9 menghanguskan seluruh run. Colab free
memutus sesi jauh sebelum cap 12 jam; `@app.function(timeout=24*3600)` tidak.

**Kamu dapat bf16.** `train_fga.py` menjaga bobot di float32 dan menyerahkan
presisi campuran ke `autocast`, karena FGA berisi softmax attention dan LayerNorm
yang tidak stabil di fp16. Colab free memberi T4 (Turing) yang **tidak punya bf16
native**, sehingga notebook jatuh ke fp16 — risiko nyata, bukan teoretis. Di sini
GPU dipilih eksplisit: `GPU_TRAIN = "L4"` (Ada) punya bf16.

**Volume, bukan Drive.** Cache latent dan checkpoint hidup di Modal Volume. Tidak
ada `drive.mount()`, tidak ada kuota Drive, dan tidak ada variabel Python yang
hilang setelah restart runtime (penyebab error `--data_dir: expected one argument`
yang punya sel khusus di notebook).

Yang **hilang** dibanding notebook: gate visual. Di Modal tidak ada
`plt.imshow()`, jadi `gate_roundtrip` menulis PNG ke Volume untuk kamu ambil.
Jangan lewati — GATE 1b adalah yang menangkap salah konvensi scaling latent.

---

## 2. Biaya

Trainingnya ringan: hanya `AutoencoderKL` yang dimuat, satu micro-step =
`vae.decode()` + loss. Dengan `--iters 10000 --accum 4` (40.000 micro-step) pada
crop 256², perkiraannya ~3-5 jam di T4 dan ~2 jam di L4.

| Fase | GPU | Perkiraan durasi | Perkiraan biaya |
|---|---|---|---|
| `make_pairs` (100 citra x 4 draws) | T4 | ~10 menit | < $0.15 |
| `cache` (400 pasang, 1 step) | L4 | ~15 menit | < $0.25 |
| `train` (satu varian) | L4 | ~2 jam | ~$1.60 |
| `infer` + `metrics` (3 kondisi) | L4 | ~15 menit | < $0.25 |
| **Total, partial + full** | | **~5 jam** | **~$4** |

Tier GPU bisa diganti lewat `IFGA_GPU` (lihat §9), tetapi baca dulu alasan
mengapa itu jarang menolong untuk beban kerja ini — A100 dengan `--batch 1`
berjalan hampir sama cepatnya dengan L4 pada harga 2,5 kali.

Kredit free tier Modal (kira-kira $30/bulan pada saat dokumen ini ditulis) muat
untuk seluruh eksperimen berkali-kali, termasuk probe `w_freq` di §9.
**Verifikasi tarif dan besaran kredit di [modal.com/pricing](https://modal.com/pricing)
sebelum mengandalkan angka di atas** — keduanya bisa berubah, dan penyimpanan
Volume ditagih terpisah dari kredit compute.

---

## 3. Persiapan sekali jalan

```bash
uv tool install modal
modal setup
```

**Jangan pakai `pip install modal`** bila Python-mu dari Homebrew: ia ditandai
*externally managed* (PEP 668), jadi pip menolak dengan
`error: externally-managed-environment` — dan karena pesan itu tenggelam di
belasan baris saran, instalasinya tampak berhasil padahal tidak ada apa pun yang
terpasang, sehingga `modal` berakhir sebagai `command not found`.
`--break-system-packages` memang bisa menembusnya, tetapi itu menulis ke Python
milik Homebrew dan bisa rusak saat `brew upgrade`.

Modal 1.5.5 menerima Python `>=3.10,<3.15`.

`modal setup` membuka browser untuk autentikasi dan menulis token ke
`~/.modal.toml`. Tidak ada yang perlu disiapkan di sisi Volume — ia dibuat
otomatis oleh `modal.Volume.from_name(..., create_if_missing=True)`.

Verifikasi:

```bash
modal volume list
```

### Kode lokal dipakai apa adanya

`modal_train.py` memount direktori kerjamu dengan `add_local_dir(".", "/repo")`
dan `copy=False`, jadi:

- **Tidak perlu commit/push** sebelum menjalankan. Yang dijalankan adalah kode di
  disk mesinmu, termasuk perubahan yang belum di-commit.
- Mengedit `losses.py` **tidak** memicu rebuild image.

Karena `add_local_dir` harus jadi lapisan **terakhir**, setiap varian image
diturunkan dari `_base` lewat helper `_with_repo()`. Menambahkan `pip_install`
atau `run_commands` setelah berkas lokal ditempelkan akan ditolak Modal dengan
`An image tried to run a build step after using image.add_local_*`.

### `pip install -e .` sengaja dilewati

`setup.py` di repo ini adalah milik diffusers, dan satu-satunya gunanya di
notebook adalah membuat `basicsr` ditemukan. Itu sudah dikerjakan oleh
`PYTHONPATH=/repo:/repo/src` di definisi image. Urutan itu penting: `src/` harus
mendahului agar diffusers yang di-vendor (yang menyediakan
`StableDiffusionInvEnhancePipeline` dan `NoisePredictor`) tidak ter-shadow paket
PyPI.

### `albumentations` WAJIB dipasang

Notebook Colab menyebut `albumentations` sebagai "demo/eval only" dan
melewatinya. **Itu keliru** — notebook hanya lolos karena Colab sudah memuatnya
lebih dulu. Kenyataannya:

```
sampler_invsr.py -> datapipe.datasets -> basicsr.data.realesrgan_dataset
                                          `-- import albumentations  (level modul)
```

Jadi ia dibutuhkan bahkan untuk caching dan inferensi, bukan hanya `make_pairs`.
Sudah masuk daftar `pip_install` di `modal_train.py`; jangan hapus.

---

## 4. Peta fase

| Fase | Perintah | GPU |
|---|---|---|
| 1 | `modal run modal_train.py::prepare` | — |
| 2 | `modal run modal_train.py::make_pairs` | T4 |
| 3-4 | `modal run modal_train.py::cache` | L4 |
| GATE 1b | `modal run modal_train.py::gate_roundtrip` | L4 |
| GATE 2 | `modal run modal_train.py::gate2` | L4 |
| 5 | `modal run modal_train.py::train --iters 50 --accum 2 --log-every 5 --val-every 25` | L4 |
| 6-7 | `modal run modal_train.py::train --mode partial` | L4 |
| 8 | `modal run modal_train.py::infer` | L4 |
| GATE 3 | `modal run modal_train.py::gate_diff` | — |
| 9 | `modal run modal_train.py::metrics` | L4 |

Semua state ada di Volume `ifga-sr`:

```
/vol/
├── source_hr/                citra HR DIV2K
├── pairs/{lr,gt}/            pasangan hasil degradasi Real-ESRGAN
├── cache/steps1/{latent,gt}/ keluaran cache_latents.py
├── configs/                  modal-sd-turbo.yaml (dibuat otomatis)
├── weights/                  noise_predictor_sd_turbo_v5.pth
├── models/                   cache HF (sd-turbo)
├── experiments/fga_*/        checkpoint + config.json + history.json
├── out/                      hasil inferensi + log metrik
└── gates/                    PNG bukti gate
```

---

## 5. Fase 1 — Citra HR sumber

```bash
modal run modal_train.py::prepare
```

Default `valid` = 100 citra / 449 MB. Untuk run yang dilaporkan pakai `train`
(800 citra / 3.5 GB):

```bash
modal run modal_train.py::prepare --dataset train
```

Idempoten: kalau `/vol/source_hr` sudah berisi PNG, ia langsung keluar.

Unduhannya memakai `requests` dengan streaming, bukan `wget` — `debian_slim`
tidak memuat wget, dan menambahkannya ke `apt_install` akan membatalkan cache
lapisan pip di atasnya sehingga torch ikut ter-build ulang. Ukuran unduhan
dicocokkan dengan `content-length`, karena zip yang terpotong gagal dengan pesan
yang membingungkan. Progres dicetak tiap ~134 MB; untuk `--dataset train`
(3,5 GB) hitung sekitar 3-6 menit.

Untuk memakai citra sendiri (LSDIR, FFHQ), unggah dulu lalu lewati fase ini:

```bash
modal volume put ifga-sr ./citra_hr_lokal /source_hr
```

---

## 6. Fase 2 — Pasangan LR/GT

Uji 8 citra dulu:

```bash
modal run modal_train.py::make_pairs --limit 8
```

Lalu set penuh:

```bash
modal run modal_train.py::make_pairs
```

### Ukuran HR: pilih sekarang, tidak bisa diubah nanti

Default di sini **`gt_size=512`**, berbeda dari 256 di notebook. Alasannya
konkret: `--crop` pada `train_fga.py` **dilewati** bila ukuran crop >= citra.
Dengan cache 256², `--crop 256` adalah no-op. Dengan GT 512² dan `--crop 256`,
kamu mendapat crop acak yang benar-benar bervariasi — biaya per micro-step tetap
di level 256², tetapi ragam datanya naik banyak.

Mengecilkannya nanti **mewajibkan caching ulang.** Putuskan sebelum §7.

`--draws 4` menghasilkan 4 undian degradasi independen per citra HR. Ini penting:
caching membekukan satu degradasi per berkas, sedangkan InvSR melihat undian baru
setiap epoch. Beberapa undian memulihkan sebagian ragam itu.

Periksa hasilnya dengan mata:

```bash
modal volume get ifga-sr /pairs/lr ./cek_pairs
```

LR harus terlihat benar-benar terdegradasi (blur + noise + artefak JPEG), bukan
sekadar berukuran kecil.

---

## 7. Fase 3-4 — Cache latent + GATE 1b

Smoke test 4 citra:

```bash
modal run modal_train.py::cache --limit 4
```

Skrip akan `raise RuntimeError` bila bentuk latent bukan `(1, 4, H/8, W/8)` —
itu penjaga terhadap pipeline yang mengabaikan `output_type="latent"`.

Fungsi ini juga menulis `configs/modal-sd-turbo.yaml` ke Volume, yang menyelesaikan
tiga masalah `configs/sample-sd-turbo.yaml`:

1. `model_start.ckpt_path: ~` → `sampler_invsr.py` assert gagal. Bobot noise
   predictor diunduh dari HF dan disimpan ke Volume.
2. `cache_dir` menunjuk path cluster penulis asli.
3. **`timesteps` default caching `[250, ...]` berbeda dari jadwal inferensi
   `[200]`.** Melatih di satu jadwal lalu mengevaluasi di jadwal lain adalah
   train/test mismatch **di dalam bagian pipeline yang dibekukan** — angkanya
   akan terlihat masuk akal tetapi tidak berarti seperti yang kamu pikir.
   Config yang ditulis di sini memakai jadwal inferensi untuk keduanya.

### GATE 1b — konvensi scaling latent

```bash
modal run modal_train.py::gate_roundtrip
modal volume get ifga-sr /gates/roundtrip.png ./roundtrip.png
```

`cache_latents.py` menyimpan latent **mentah**, sedangkan `train_fga.py`
membaginya dengan `vae.config.scaling_factor` sebelum decode. Kalau konvensi ini
tidak cocok, FGA akan dilatih untuk mengompensasi bug, bukan untuk menambah
detail.

**Lulus:** gambar terlihat wajar. Noise abu-abu atau distorsi berat = berhenti
dan perbaiki.

Cache penuh:

```bash
modal run modal_train.py::cache
```

Aman diinterupsi dan dilanjutkan — berkas yang sudah ada dilewati. **Satu
direktori cache ini dipakai bersama oleh run `partial` dan `full`.** Jangan
pernah membuatnya ulang di antara dua varian; cache bersama itulah sebagian dari
apa yang membuat ablasi terkontrol.

---

## 8. Fase 5 — GATE 2: smoke test

Ada dua bagian: pemeriksaan otomatis (yang menjawab keempat kriteria secara
langsung) dan smoke test training (yang membuktikan loop-nya benar-benar jalan).

### 8a. Pemeriksaan otomatis

```bash
modal run modal_train.py::gate2
```

Ini tidak melatih apa pun. Ia memuat VAE tiga kali — `none`, `partial`, `full` —
lalu men-decode beberapa latent dari cache dan membandingkan hasilnya. Keluarannya
berakhir dengan tabel verdict:

```
========================================================================
LULUS  | 1. parameter dilatih bukan nol             | partial=... full=...
LULUS  | 2. parameter full > partial                | ... > ...
LULUS  | 3. zero-init: keluaran identik baseline    | max|selisih| ... (harus tepat 0.0)
LULUS  | 4. loss finit                              | tidak ada nan/inf
========================================================================
```

Fungsinya `assert` di akhir, jadi `modal run` keluar dengan status non-nol bila
ada yang gagal — aman dipakai sebagai gerbang di skrip.

**Kenapa perlu fungsi terpisah, bukan sekadar membaca log training.** Kriteria 3
berbunyi "loss di step 0 setara loss rekonstruksi baseline", tetapi tidak ada
apa pun di alur ini yang menghitung angka baseline itu, dan baris log pertama
`train_fga.py` baru muncul di step ke-`log_every` — bukan step 0. `gate2`
menghitung baselinenya eksplisit dengan `mode="none"`.

Pemeriksaannya juga lebih ketat daripada "setara". Karena `unembed` di-zero-init,
`delta = conv(bobot 0, bias 0)` bernilai **tepat** 0.0, sehingga keluaran
partial/full yang belum dilatih harus **identik bit** dengan baseline — bukan
sekadar mirip. Karena itu `gate2` berjalan di float32 tanpa `autocast`, dan
kriteria lulusnya `max|rec - baseline| == 0.0`. Nilai bukan-nol berarti zero-init
tidak berlaku dan cabang residual sudah merusak prior pretrained sejak iterasi
pertama.

### 8b. Smoke test training

```bash
modal run modal_train.py::train --mode partial --iters 50 --accum 2 \
  --log-every 5 --val-every 25 --tag smoke
modal run modal_train.py::train --mode full --iters 50 --accum 2 \
  --log-every 5 --val-every 25 --tag smoke
```

`--log-every` dan `--val-every` **wajib diturunkan** untuk run 50 iterasi. Default
fungsinya (50 dan 500) dibuat untuk run panjang: pada 50 iterasi kamu hanya
mendapat satu baris log dan **validasi tidak pernah berjalan sama sekali**,
sehingga tidak ada `history.json` yang berguna.

Yang dibaca dari log:

| Kriteria | Baris yang dicari | Lulus bila |
|---|---|---|
| 1 & 2 | `[model] mode=... \| parameter dilatih = N (x% dari VAE)` | N bukan nol, dan N pada `full` > N pada `partial` |
| 3 | baris `[5/50] ... loss_total=...` pertama | setara nilai baseline dari §8a; bukan lonjakan besar |
| 4 | setiap baris `loss_*=` | tidak ada `nan` atau `inf` |

Ditambah dua hal yang hanya bisa dilihat di sini: baris `-> VAL loss=... psnr=...
lpips=...` benar-benar muncul (artinya jalur validasi dan pemilihan checkpoint
hidup), dan berkas `fga_partial_smoke_best.pth` tertulis ke Volume.

Kalau muncul `nan` di L4, itu bukan masalah bf16 — periksa `--w-freq`. Kalau kamu
memindahkannya ke GPU Turing, `--amp off` adalah jalan keluarnya, bukan `bf16`.

### 8c. Proyeksi biaya sebelum run panjang

Baca `[selesai] N menit` dari smoke test, lalu:

```
menit_per_micro = N / (50 * 2)
jam_run_penuh   = menit_per_micro * iters * accum / 60
```

Untuk `--iters 10000 --accum 4` (40.000 micro-step), pastikan hasilnya di bawah
timeout function (24 jam). Kalau terlalu lama, turunkan `--inner-dim` ke 32 —
itu kira-kira menyeparuh komputasi FGA — atau kecilkan `--crop`.

### 8d. Bersihkan artefak smoke test

```bash
modal volume rm -r ifga-sr /experiments/fga_partial_smoke
modal volume rm -r ifga-sr /experiments/fga_full_smoke
```

## 9. Fase 6-7 — Latih `partial` dan `full`

```bash
modal run --detach modal_train.py::train --mode partial
modal run --detach modal_train.py::train --mode full
```

**Pakai `--detach`.** Tanpanya, menutup terminal atau kehilangan koneksi akan
membunuh run — dan `train_fga.py` tidak punya resume. Dengan `--detach`, run
tetap hidup di server; pantau dari dashboard atau `modal app logs`.

### Default fungsi ini bukan default skrip

`train` di `modal_train.py` memakai `freq_mode="magnitude"`,
`select_by="lpips"`, `w_freq=1.0`, `w_lpips=0.5`. Ini **disengaja** dan berbeda
dari default `train_fga.py`.

Default skripnya (`freq_mode="full"`, `select_by="psnr"`) menghasilkan model yang
lebih **halus** daripada baseline:

- `frequency_l1_loss` menghitung selisih FFT **kompleks** pada basis ortonormal.
  Karena rFFT ortonormal adalah transformasi uniter, teorema Parseval membuatnya
  ekuivalen dengan loss piksel — bukan sinyal supervisi baru. Ia sensitif fase,
  jadi tekstur yang benar secara statistik tetapi bergeser beberapa piksel
  dihukum berat. Cara termurah bagi optimizer untuk menurunkannya adalah
  **meredam** detail tersebut.
- `--select_by psnr` memilih checkpoint dengan error kuadrat terkecil, yaitu
  varian paling halus di antara semua step tervalidasi.

`freq_mode="magnitude"` membuang fase lewat `.abs()` sebelum menghitung selisih,
jadi yang disupervisi adalah energi tiap frekuensi. Begitu energi band tinggi
turun di bawah GT, loss **naik** — penghalusan tidak lagi jadi jalan keluar murah.

### Kondisi pembanding untuk tabel ablasi

Kamu tetap membutuhkan angka dari objective lama. Jalankan dengan `tag` berbeda
supaya tidak menimpa:

```bash
modal run --detach modal_train.py::train --mode partial \
  --freq-mode full --w-freq 0.1 --w-lpips 0.0 --select-by psnr --tag base
```

Perhatikan konversi nama flag: Modal CLI memetakan parameter Python `w_freq`
menjadi `--w-freq` (garis bawah jadi tanda hubung).

### Mengganti GPU — dan kenapa itu jarang menolong

GPU dibaca dari environment saat import, karena `gpu=` pada `@app.function`
dievaluasi saat dekorasi, bukan saat pemanggilan:

```bash
IFGA_GPU=A100-40GB modal run --detach modal_train.py::train --mode partial
```

Nilai yang umum tersedia: `T4`, `L4`, `A10G`, `L40S`, `A100-40GB`, `A100-80GB`,
`H100`. Cek dokumentasi Modal untuk daftar terkini — ketersediaan dan namanya
berubah. Semua dari A10G ke atas punya bf16 native.

**Tetapi naik tier kemungkinan besar tidak mempercepat banyak.** Pada `--batch 1`
dengan crop 256, satu micro-step hanya men-decode latent 32x32x4 menjadi
256x256x3. Ditambah tiga hal:

- `vae.decoder.gradient_checkpointing = True` menukar komputasi dengan memori —
  aktivasi dihitung ulang, jadi masalahnya bukan kapasitas memori.
- FGA dipakai dengan `window_size=1, overlap_ratio=4`, sehingga menghasilkan
  `h*w` window attention yang masing-masing hanya 4 query x 25 key. Ribuan
  operasi mungil seperti ini dibatasi **overhead kernel dan bandwidth**, bukan
  FLOPs.
- `--batch 1` membuat okupansi GPU sangat rendah. H100 tidak memperbaiki okupansi
  rendah; ia hanya menganggur lebih mahal.

Jadi H100 untuk model 0,3 juta parameter di batch 1 adalah kredit yang terbakar
sia-sia.

**Pengungkit yang benar: naikkan `--batch`.** Ini baru mungkin karena `--crop`
membuat semua sampel berukuran seragam — tanpa crop, dataset mengembalikan citra
penuh dan `default_collate` tidak bisa menumpuk tensor beda ukuran (itu sebabnya
notebook mewajibkan `--batch 1`).

Jaga `batch x accum` tetap sama agar batch efektifnya tidak berubah:

```bash
# ekuivalen secara matematis dengan --batch 1 --accum 4
IFGA_GPU=A100-40GB modal run --detach modal_train.py::train \
  --mode partial --batch 4 --accum 1
```

Gradiennya identik: `train_fga.py` memakai `loss / args.accum` dan loss-nya
sendiri rata-rata atas batch, jadi rata-rata atas 4 sampel adalah hal yang sama
entah lewat batch atau akumulasi.

**Ukur dulu, jangan asumsi.** Jalankan smoke test §8b di L4, lalu ulangi di tier
yang kamu incar dengan `--batch` yang lebih besar, dan bandingkan
`[selesai] N menit`. Kalau selisihnya kecil, GPU bukan hambatanmu.

Dua pengungkit lain yang berpengaruh lebih besar daripada tier GPU:
`--inner-dim 32` (kira-kira menyeparuh komputasi FGA) dan `--crop` yang lebih
kecil.

> **Syarat ablasi.** `batch`, `accum`, dan GPU harus **identik** antara `partial`
> dan `full`. Melatih satu varian di A100 batch 4 dan satunya di L4 batch 1
> membuat perbandingan H3 terkonfound, meski batch efektifnya sama.

### Flag yang tersedia di `train`

Setiap argumen loss `train_fga.py` diteruskan oleh fungsi Modal, dengan garis
bawah menjadi tanda hubung:

| Modal CLI | default | keterangan |
|---|---|---|
| `--w-pixel` | 1.0 | penyumbang utama tarikan ke arah blur; turunkan ke 0.1 |
| `--w-freq` | 1.0 | bobot term frekuensi |
| `--w-lpips` | 0.5 | bobot LPIPS |
| `--lpips-net` | alex | **pakai `vgg`** — alex terpuaskan oleh penghalusan |
| `--freq-mode` | magnitude | `full` \| `highpass` \| `magnitude` |
| `--freq-cutoff` | 0.25 | batas band, fraksi Nyquist |
| `--pixel-type` | l1 | `l1` \| `l2` |
| `--select-by` | lpips | `psnr` \| `lpips` \| `loss` |
| `--lr` `--amp` `--seed` `--val-size` | 1e-4, bf16, 123456, 32 | |
| `--crop` `--batch` `--accum` `--inner-dim` | 256, 1, 4, 64 | |
| `--log-every` `--val-every` | 50, 500 | turunkan untuk smoke test |

### Kalibrasi `w_freq`

Skala `magnitude` berbeda dari `full`, jadi `w_freq=0.1` terlalu lemah untuk
mengalahkan tarikan L1 piksel. Lakukan probe 500 iterasi sebelum run panjang:

```bash
for W in 0.5 1.0 2.0; do
  modal run modal_train.py::train --mode partial --iters 500 \
    --w-freq $W --tag "probe$W"
done
```

### Yang menandakan training sehat — bukan PSNR naik

Dengan objective ini, **PSNR turun sedikit sementara LPIPS jelas membaik**. Itu
tanda modul frekuensi bekerja, bukan regresi. Kalau PSNR ikut naik, `w_freq`
masih terlalu kecil.

- `loss_freq` harus turun **bersamaan** dengan `loss_pixel`, tidak saling tukar.
- `spec` (konsistensi spektral) naik.
- Kalau PSNR validasi **persis** di nilai baseline selama ribuan step, FGA
  belajar mengeluarkan nol — cabang residual tidak menemukan sinyal. Periksa LR
  dan pastikan gradien sampai ke parameter FGA.

### Syarat perbandingan terkontrol

`--data_dir`, `--iters`, `--batch`, `--accum`, `--crop`, `--lr`, `--seed`,
`--amp`, dan **setiap bobot loss** harus identik antara `partial` dan `full`.
Satu saja berbeda, perbandingannya terkonfound dan tidak ada kesimpulan H3 yang
mengikutinya. Verifikasi setelah keduanya selesai:

```bash
modal volume get ifga-sr /experiments/fga_partial_mag/config.json ./p.json
modal volume get ifga-sr /experiments/fga_full_mag/config.json ./f.json
python -c "
import json
a, b = json.load(open('p.json')), json.load(open('f.json'))
diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
extra = set(diff) - {'mode', 'out_dir'}
print(json.dumps(diff, indent=2))
print('TERKONFOUND:', extra if extra else 'tidak — perbandingan valid')
"
```

---

## 10. Fase 8-9 — Inferensi, GATE 3, metrik

Tiga kondisi, perintah identik kecuali `--fga-mode`:

```bash
modal run modal_train.py::infer --fga-mode none
modal run modal_train.py::infer --fga-mode partial --tag mag
modal run modal_train.py::infer --fga-mode full    --tag mag
```

`infer` merekonstruksi split validasi yang sama dengan `LatentHRDataset`
(`val_size` nama pertama setelah disortir), jadi FGA tidak pernah melihatnya saat
training. Melaporkan angka di pasangan training tidak bermakna.

### GATE 3 — keluaran harus benar-benar berbeda dari baseline

```bash
modal run modal_train.py::gate_diff
```

Ini pemeriksaan paling penting di seluruh runbook. Kalau `mean abs diff` tepat
`0.0`, bobot FGA tidak sampai ke model dan **setiap metrik hilirnya hanya duplikat
baseline** — hasil yang terlihat seperti "FGA tidak berpengaruh" padahal itu bug
wiring. `partial` dan `full` juga harus berbeda satu sama lain, kalau tidak dua
kondisimu diam-diam model yang sama.

Metrik:

```bash
modal run modal_train.py::metrics --name baseline
modal run modal_train.py::metrics --name partial_mag
modal run modal_train.py::metrics --name full_mag
```

PSNR/SSIM menghargai fidelitas piksel dan cenderung **menghukum** detail
frekuensi tinggi yang justru ingin ditambahkan FGA. Laporkan LPIPS di
sampingnya, jangan PSNR sendirian: penurunan PSNR kecil dengan perbaikan LPIPS
yang jelas adalah tanda modul frekuensi bekerja, bukan regresi.

Untuk mereproduksi angka kuantitatif protokol InvSR pada ImageNet-Test dan
RealSRV3, tambahkan `--color-fix wavelet`.

### Dua flag yang tidak boleh dilupakan

`inference_invsr.py` **menimpa** `cache_dir` dan `model_start.ckpt_path` dari YAML
dengan defaultnya sendiri, `./weights` — relatif terhadap `/repo`, yang read-only
di container. Tanpa `--sd_path` dan `--started_ckpt_path` yang menunjuk Volume, ia
mencoba mengunduh ulang sd-turbo ~2,5 GB setiap run atau langsung gagal menulis.
`modal_train.py::infer` sudah mengirim keduanya; ingat ini bila kamu memanggil
`inference_invsr.py` secara manual lewat `modal shell`.

### Catatan presisi

`configs/sample-sd-turbo.yaml` memuat pipeline dengan
`torch_dtype: torch.float16`, jadi jalur upsampler bawaan berjalan fp16 saat
inferensi sementara FGA dilatih di float32. `inject_fga` menjaga FGA tetap
float32 dan mengonversi di batas modul, tetapi selisih presisi jalur `base`
tetap ada. Ukur dan laporkan, jangan diasumsikan nol.

Config yang sama juga menyalakan `tiled_vae: True` dengan
`latent_tiled_size: 128`. Untuk evaluasi 256²-512² tiling tidak aktif. Kalau kamu
mengevaluasi citra > 1024 px, decoder berjalan per-tile sementara training tidak
pernah begitu, dan `delta` FGA ikut ter-blend di sambungan tile — sebutkan itu
bila melaporkan angka pada RealSet80.

---

## 10a. Set evaluasi: cacat split dan cara menghindarinya

**Cacat yang ditemukan pada 2026-09-09.** `LatentHRDataset` versi lama memilih
validasi sebagai **32 nama pertama setelah disortir**. Karena `make_pairs.py`
menamai berkas `<scene>_d<draw>`, batas itu bergeser setiap kali `--draws`
berubah:

| `--draws` | 32 nama pertama | scene unik |
|---|---|---|
| 4 | 0801–0808 x d0–d3 | 8 |
| 8 | 0801–0804 x d0–d7 | **4** |

Hanya 16 nama beririsan. Konsekuensinya berlapis:

- Hasil sebelum dan sesudah perubahan `--draws` **dievaluasi pada gambar yang
  berbeda**, jadi tidak sebanding.
- Scene 0805–0808 berpindah dari validasi ke **training**, sehingga checkpoint
  lama dan baru bahkan tidak berbagi definisi "data yang belum dilihat".
- Keragaman scene turun dari 8 ke 4.

**Mengapa ini fatal untuk metrik ketajaman.** Varians Laplacian GT terukur
0.019818 pada subset 8-scene dan 0.007133 pada subset 4-scene — berbeda **2,8x**.
Metrik ketajaman didominasi konten scene, bukan model. Pada subset lama baseline
tampak 29% **di bawah** GT; pada subset baru ia 46% **di atas** GT. Kesimpulan
yang berlawanan dari model yang sama.

### Perbaikannya

`--split-by scene` (sekarang default) menahan scene utuh beserta seluruh
draw-nya, jadi menambah undian degradasi tidak lagi menggeser batas train/val:

```bash
modal run --detach modal_train.py::train --mode partial \
  --split-by scene --val-scenes 8 --tag v3
```

`--split-by name` masih ada, khusus untuk mereproduksi run lama. Ia mencetak
peringatan, dan nilainya terekam di `config.json`.

### Set evaluasi eksplisit

Untuk perbandingan yang harus tahan terhadap perubahan data apa pun, sebut
scene-nya secara langsung — ini mengabaikan seluruh logika split:

```bash
modal run modal_train.py::infer --fga-mode partial --tag v2 \
  --scenes 0801,0802,0803,0804 --out-name partial_v2_fixed
```

`--per-scene N` mengambil N draw pertama tiap scene. Untuk metrik ketajaman,
**banyak scene x sedikit draw** jauh lebih informatif daripada sedikit scene x
banyak draw pada jumlah gambar yang sama.

> **Aturan.** Saat membandingkan checkpoint yang dilatih pada split berbeda, set
> evaluasi harus berisi hanya scene yang ditahan dari **semua** checkpoint yang
> dibandingkan. Untuk checkpoint di repo ini, irisan itu adalah 0801–0804.

---

## 10b. Menghindari hasil yang tertimpa

Tidak ada satu pun skrip di repo ini yang memperingatkan saat menimpa:
`inference_invsr.py` memakai `mkdir(delete=False)` lalu menulis PNG dengan nama
berkas input yang sama, dan `save_fga` menimpa `.pth` apa adanya. Dua kali jalan
dengan nama sama = hasil lama hilang tanpa jejak.

Tiga lapis perlindungan:

**1. Nama otomatis menyertakan rezim sampling.** `infer` menulis ke
`out/baseline_s5`, `out/partial_v2_s5` — jadi hasil 1-step dan 5-step tidak
pernah bertabrakan.

**2. `--tag` memisahkan varian training.** Satu tag = satu direktori
`experiments/fga_<mode>_<tag>/`. Ganti tag setiap kali komposisi loss berubah;
itu juga yang membuat tabel ablasimu bisa dilacak.

**3. Penjaga eksplisit.** `train`, `infer`, dan `metrics` menolak berjalan bila
targetnya sudah berisi hasil:

```
RuntimeError: /vol/out/partial_v2_s5 sudah berisi 32 berkas *.png (mis. 0801_d0.png).
Menjalankan ini akan menimpanya. Pilih salah satu:
  - pakai --out-name <nama lain>, atau --tag yang berbeda
  - tambahkan --overwrite bila memang ingin menimpa
```

Lewati dengan `--overwrite` hanya saat kamu memang ingin mengganti hasil itu —
misalnya mengulang run yang gagal di tengah.

`infer` juga punya `--out-name` untuk menamai direktori keluaran secara bebas,
mis. `--out-name partial_v2_s5_wavelet` saat membandingkan efek `--color-fix`
pada checkpoint yang sama.

---

## 11. Mengambil hasil

```bash
# checkpoint + log (yang kamu arsipkan bersama angka yang dilaporkan)
modal volume get ifga-sr /experiments ./experiments

# citra hasil + log metrik
modal volume get ifga-sr /out ./out
```

Lima artefak yang membuat hasilmu reproducible: `config.json`, `history.json`,
berkas `.pth`, jumlah parameter dari GATE 2, dan `mean abs diff` dari GATE 3.

---

## 12. Troubleshooting

| Gejala | Sebab | Perbaikan |
|---|---|---|
| `ModuleNotFoundError: albumentations` | notebook menyebutnya opsional; sebenarnya diimpor di level modul oleh `realesrgan_dataset` | sudah ada di `pip_install`; jangan hapus |
| `ImportError: cannot import name 'StableDiffusionInvEnhancePipeline'` | diffusers PyPI menaungi versi vendored di `src/` | pastikan `PYTHONPATH` menempatkan `/repo/src` lebih dulu, dan **jangan** `pip install diffusers` |
| `libGL.so.1: cannot open shared object file` | `cv2` butuh pustaka sistem | `apt_install("libgl1", "libglib2.0-0")` — sudah ada |
| `AssertionError` pada `ckpt_path is not None` | `sample-sd-turbo.yaml` mengirim `ckpt_path: ~` | jalankan lewat `cache`/`infer`, yang memanggil `_write_config` |
| Checkpoint hilang setelah function selesai | `vol.commit()` tidak dipanggil | setiap function di `modal_train.py` sudah memanggilnya; tambahkan bila kamu menulis function baru |
| sd-turbo diunduh ulang setiap run | `HF_HOME` tidak menunjuk Volume | sudah diset di `.env()` pada image |
| Inferensi mengunduh ulang sd-turbo, atau gagal tulis di `./weights` | `inference_invsr.py` menimpa `cache_dir`/`ckpt_path` dari YAML | kirim `--sd_path` dan `--started_ckpt_path` (lihat §10) |
| `add_local_dir` tidak dikenali | CLI Modal terlalu lama | `uv tool install modal` (lihat §3) |
| `An image tried to run a build step after using image.add_local_*` | ada `pip_install`/`run_commands` setelah lapisan berkas lokal | turunkan varian image dari `_base`, dan panggil `_with_repo()` paling akhir |
| `error: externally-managed-environment` saat `pip install modal` | Python Homebrew ditandai PEP 668, jadi pip menolak dan tidak memasang apa pun | `uv tool install modal` — jangan `--break-system-packages` |
| `[Errno 2] No such file or directory: 'wget'` | `debian_slim` tidak memuat wget | sudah diganti `requests`; jangan tambahkan wget ke `apt_install` — itu membatalkan cache lapisan pip dan memaksa rebuild torch ~2,5 GB |
| `OSError: [Errno 18] Invalid cross-device link` | `os.rename` dari `/tmp` ke `/vol`, dua filesystem berbeda | tulis langsung ke Volume, atau pakai `shutil.move` |
| Run mati saat terminal ditutup | `--detach` tidak dipakai | `modal run --detach ...` |
| `nan` pada loss | fp16 pada GPU non-Ampere | tetap di L4 (bf16), atau `--amp off` |
| Jumlah parameter `partial` == `full` | pemilihan mode rusak | periksa `inject_fga`; ablasi tidak valid sampai ini beres |

### Memantau run yang di-detach

```bash
modal app list
modal app logs <app-id>
```

### Masuk ke container untuk investigasi

```bash
modal shell modal_train.py::train
```

Memberi shell di image yang sama dengan `/repo` dan `/vol` terpasang — cara
tercepat memeriksa kenapa sebuah impor gagal.

---

## 13. Checklist

- [ ] `modal setup` selesai, `modal volume list` menampilkan `ifga-sr`
- [ ] Fase 1: citra HR ada di `/vol/source_hr`
- [ ] Fase 2: `--limit 8` diperiksa dengan mata, lalu set penuh dibuat
- [ ] Ukuran GT diputuskan (**512** bila ingin `--crop 256` berfungsi)
- [ ] Fase 3: smoke cache lolos, bentuk latent tercetak benar
- [ ] GATE 1b: `roundtrip.png` terlihat wajar
- [ ] Fase 4: cache penuh selesai, jumlah latent == jumlah GT
- [ ] GATE 2 otomatis: `gate2` menampilkan LULUS untuk keempat kriteria
- [ ] GATE 2 smoke test: baris `-> VAL` muncul, checkpoint tertulis, tidak ada `nan`
- [ ] Jumlah parameter `partial` dan `full` dicatat untuk tabel hasil
- [ ] `w_freq` dikalibrasi lewat probe 500 iterasi
- [ ] Fase 6: `partial` selesai (`--detach`)
- [ ] Fase 7: `full` selesai dengan flag identik kecuali `--mode`
- [ ] `config.json` kedua varian hanya beda `mode` dan `out_dir`
- [ ] Kondisi pembanding (`--tag base`) juga dilatih untuk tabel ablasi
- [ ] Fase 8: tiga kondisi inferensi dijalankan
- [ ] GATE 3: `mean abs diff` bukan nol untuk keduanya, dan `partial != full`
- [ ] Fase 9: metrik ketiga kondisi terkumpul, LPIPS dilaporkan bersama PSNR
- [ ] Artefak diambil ke lokal dan diarsipkan
