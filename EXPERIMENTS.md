# Catatan Eksperimen — Perintah Training per Versi

Riwayat lengkap konfigurasi training FGA, perintah persisnya, dan hasil yang
terukur. Setiap angka di sini berasal dari pengukuran pada gambar identik;
kelompok yang tidak sebanding ditandai eksplisit.

Dokumen pendamping: [`TRAINING.md`](TRAINING.md) (desain dan alasan tiap loss),
[`MODAL.md`](MODAL.md) (cara menjalankan).

---

## Ringkasan

| Tag | Dataset | Split | Objective | Hasil singkat |
|---|---|---|---|---|
| `base` | DIV2K | `name` | fidelitas awal | 0.04× ketajaman GT — runtuh |
| `mag` | DIV2K | `name` | loss magnitudo | 0.63× GT |
| `v2` | DIV2K | `name` | + LPIPS-VGG, piksel turun | 0.69× GT, **menang di semua metrik referensi** |
| `v3` | DIV2K | `scene` 20 | + loss ketajaman + GAN | 1.52× GT, +38% dari baseline, **ada bercak chroma** |
| `v4` | LSDIR+FFHQ | `scene` 20 | sama seperti `v3` | chroma 12.9× GT — noise warna parah |
| `v5` | LSDIR+FFHQ | `scene` 20 | + penjaga chroma, asimetris, TTUR | chroma bersih, tapi 2.65× GT — terlalu tajam |
| `v6` | LSDIR+FFHQ | `scene` 20 | kembali ke fidelitas | −50% dari baseline — halus, chroma bersih |
| `v7` | LSDIR+FFHQ | `scene` 20 | target 1.4× + patch-wise | *belum dijalankan* |

**Parameter konstan di semua run:**

```
--iters 10000 --accum 4 --batch 1 --crop 256
--lr 1e-4 --warmup 500 --amp bf16 --seed 123456 --inner-dim 64
```

Setiap varian `full` identik dengan `partial` kecuali `--mode full`.

---

## Kelompok yang sebanding

Tiga perubahan dataset/split memisahkan run-run ini. **Membandingkan lintas
kelompok tidak sah** — pernah terjadi di proyek ini dan menghasilkan kesimpulan
yang berbalik tanda.

| Kelompok | Tag | Dataset | Set evaluasi |
|---|---|---|---|
| A | `base`, `mag`, `v2` | DIV2K | 4–8 scene (rapuh) |
| B | `v3` | DIV2K | 20 scene |
| C | `v4`, `v5`, `v6`, `v7` | LSDIR+FFHQ | 20 scene |

Hanya **kelompok C** yang memenuhi protokol naskah (Bab 2.4.3). Angka dari
kelompok A berdiri di 16 gambar unik — terlalu sedikit untuk dilaporkan.

---

## `base` — objective awal

```bash
modal run --detach modal_train.py::train --mode partial --tag base \
  --iters 10000 --accum 4 --crop 256 --val-size 32 \
  --w-pixel 1.0 --w-freq 0.1 --freq-mode full --select-by psnr
```

Resep asli dari `TRAINING.md` versi pertama. Hasilnya **0.04× ketajaman GT** —
96% lebih tidak tajam dari foto asli.

Sebabnya struktural: setiap term adalah jarak ke GT, dan untuk SR ×4 dari
degradasi berat, peminimal jarak adalah rata-rata posterior yang blur secara
definisi. `--select-by psnr` memperburuknya dengan memilih checkpoint paling
halus dari 20 titik validasi.

## `mag` — loss magnitudo

```bash
modal run --detach modal_train.py::train --mode partial --tag mag \
  --iters 10000 --accum 4 --crop 256 --val-size 32 \
  --w-pixel 1.0 --w-freq 1.0 --freq-mode magnitude --freq-cutoff 0.25 \
  --w-lpips 0.5 --lpips-net alex --select-by lpips
```

`freq_mode` diganti dari `full` ke `magnitude`: fase dibuang sebelum selisih
dihitung, sehingga tekstur yang bergeser posisi tidak lagi dihukum. Ketajaman
naik **0.04× → 0.63× GT**, perbaikan 14 kali lipat.

## `v2` — LPIPS-VGG, bobot piksel turun

```bash
modal run --detach modal_train.py::train --mode partial --tag v2 \
  --iters 10000 --accum 4 --crop 256 --val-size 32 \
  --w-pixel 0.1 --w-freq 2.0 --freq-mode magnitude --freq-cutoff 0.5 \
  --w-lpips 2.0 --lpips-net vgg --select-by lpips
```

`alex` diganti `vgg` setelah keduanya terukur bergerak **berlawanan arah** pada
checkpoint yang sama — alex membaik sementara VGG memburuk, karena alex kurang
sensitif terhadap blur.

Hasil pada 32 gambar identik (scene 0801–0804):

| | PSNR | SSIM | LPIPS-VGG | LPIPS-ALEX | CLIPIQA | MUSIQ |
|---|---|---|---|---|---|---|
| baseline | 24.24 | 0.6350 | 0.4011 | 0.3128 | **0.7400** | **68.29** |
| `full_v2` | **24.77** | **0.6584** | **0.3752** | **0.2990** | 0.6312 | 64.73 |

**Menang di keempat metrik referensi.** Ini hasil terbersih sejauh ini: FGA
memperbaiki fidelitas rekonstruksi, tanpa perlu kualifikasi. Biayanya metrik
no-reference turun — trade-off distorsi–persepsi yang klasik.

## `v3` — loss ketajaman + adversarial

```bash
modal run --detach modal_train.py::train --mode partial --tag v3 \
  --split-by scene --val-scenes 20 \
  --w-sharp 10.0 --sharp-ratio 2.0 --w-range 1.0 \
  --w-gan 1.0 --gan-start 50 \
  --w-pixel 0.02 --w-lpips 0.1 --lpips-net vgg --w-freq 0.0 --select-by sharp
```

Pertama kali split berbasis scene dipakai, setelah split lama terbukti bergeser
saat `--draws` berubah.

Hasil pada 160 gambar dari 20 scene DIV2K:

| | ketajaman | PSNR | LPIPS-VGG | CLIPIQA | chroma/luma |
|---|---|---|---|---|---|
| baseline | 1.13× GT | 23.00 | 0.4652 | 0.6725 | 0.044 |
| `partial_v3` | 0.96× GT | 21.61 | 0.4751 | 0.5340 | 0.227 |
| `full_v3` | **1.52× GT** | 22.01 | 0.4709 | 0.6213 | 0.257 |

`full_v3` **+38% lebih tajam dari baseline pada gain 1.0** — tujuan ketajaman
tercapai dari training, bukan post-processing.

**Cacat:** rasio chroma/luma 5,8× baseline, dan kelebihannya **menggumpal** —
blok 32×32 terburuk +0.060 sementara rata-rata gambar −0.003, rasio puncak
terhadap rata-rata bermedian 13.674×. Itulah bercak hitam dan merah yang
terlihat. Bukan clipping: piksel mentok justru lebih sedikit dari baseline.

> **Catatan penting.** `v3` mendarat di 1.52× meski targetnya 2.0 karena loss
> ketajaman versi lama **jenuh di nol** begitu target terlampaui, sehingga loss
> lain menariknya turun. Ketajaman yang terlihat bagus itu sebagian hasil
> kebetulan, bukan hasil perintah.

## `v4` — konfigurasi sama, dataset mix

```bash
modal run --detach modal_train.py::train --mode partial --tag v4 --data-tag mix \
  --split-by scene --val-scenes 20 \
  --w-sharp 10.0 --sharp-ratio 2.0 --w-range 1.0 \
  --w-gan 1.0 --gan-start 50 \
  --w-pixel 0.02 --w-lpips 0.1 --lpips-net vgg --w-freq 0.0 --select-by sharp
```

Pertama kali memakai dataset sesuai naskah: 5.000 LSDIR + 5.000 FFHQ.

Gagal lebih parah dari `v3`:

| | terukur |
|---|---|
| chroma vs GT | **12.94×** |
| ketajaman luma vs GT | hanya 1.32× |
| `lap_ratio` step 500 | **47.2** |
| `loss_gan_d` akhir | 1.87 (awal hinge = 2.0) |

Tiga kegagalan bersambung: loss ketajaman mengukur luminansi saja sehingga
chroma tak terkendali, versi satu arah membiarkan lonjakan tanpa batas, dan
discriminator praktis tidak belajar sehingga penjaga yang seharusnya menolak
tekstur tidak natural sedang tidur.

## `v5` — penjaga chroma, penalti asimetris, TTUR

```bash
modal run --detach modal_train.py::train --mode partial --tag v5 --data-tag mix \
  --split-by scene --val-scenes 20 \
  --w-sharp 10.0 --sharp-ratio 2.0 --sharp-over 0.25 \
  --w-range 1.0 --w-chroma 1.0 \
  --w-gan 1.0 --gan-start 50 --d-lr 4e-4 \
  --w-pixel 0.02 --w-lpips 0.1 --lpips-net vgg --w-freq 0.0 --select-by sharp
```

Ketiga mekanismenya **berhasil**:

| sinyal | `v4` | `v5` |
|---|---|---|
| `loss_chroma` | ~12 setara | **0.06 / 0.0** |
| `lap_ratio` step 500 | 47.2 | **3.06** |
| `loss_gan_d` | 1.87 (mati) | **1.30** (belajar) |

Tetapi hasilnya lebih buruk: `lap_ratio` akhir **2.65–2.69**, PSNR turun ke
23.10/22.72, LPIPS memburuk ke 0.3603/0.3689.

Sebabnya bukan mekanisme melainkan **target**. `--sharp-ratio 2.0` dikalibrasi
saat pengukuran 4-scene menunjukkan baseline di 1.58× GT — angka yang kemudian
terbukti tidak andal. Pada 20 scene baseline hanya ~1.0–1.13× GT, jadi `v5`
dilatih untuk 2,65 kali lebih tajam dari foto aslinya. Dengan mekanisme yang
sudah benar, ia **berhasil mencapai perintah yang keliru**.

## `v6` — kembali ke objective fidelitas

```bash
modal run --detach modal_train.py::train --mode full --tag v6 --data-tag mix \
  --split-by scene --val-scenes 20 \
  --w-pixel 0.1 --w-freq 2.0 --freq-mode magnitude --freq-cutoff 0.5 \
  --w-lpips 2.0 --lpips-net vgg --w-sharp 0 --w-gan 0 --select-by lpips
```

Objective `v2` dijalankan di dataset mix — konfigurasi yang terbukti mengalahkan
baseline, kini di data yang sesuai naskah.

| | ketajaman luma | chroma | rasio C/L |
|---|---|---|---|
| baseline | — | — | 0.062 |
| `partial_v6` | −52% | −80% | 0.026 |
| `full_v6` | −50% | −81% | 0.024 |

Chroma bersih total, tanpa bercak. Tetapi setengah lebih tidak tajam dari
baseline — perilaku yang diharapkan dari objective fidelitas murni.

**Metrik referensinya belum diukur.** Itu satu-satunya yang menentukan apakah
`v6` mengulang keberhasilan `v2`.

## `v7` — target 1.4× dan statistik per-patch *(belum dijalankan)*

```bash
modal run --detach modal_train.py::train --mode full --data-tag mix --tag v7 \
  --split-by scene --val-scenes 20 \
  --w-sharp 10.0 --sharp-ratio 1.4 --sharp-over 0.25 --sharp-patch 32 \
  --w-range 1.0 --w-chroma 1.0 \
  --w-gan 1.0 --gan-start 50 --d-lr 4e-4 \
  --w-pixel 0.02 --w-lpips 0.1 --lpips-net vgg --w-freq 0.0 --select-by sharp
```

Dua perubahan dari `v5`:

**`--sharp-ratio 2.0 → 1.4`** — mereproduksi ketajaman `v3` yang disukai, kali
ini secara sengaja alih-alih sebagai akibat loss yang jenuh.

**`--sharp-patch 32`** — statistik ketajaman dan chroma dihitung per patch lalu
dirata-ratakan. Versi tingkat-gambar dapat dipenuhi dengan meledakkan beberapa
area saja; per-patch menutup celah itu.

---

## Evaluasi

Rantai standar setelah setiap training. Jalankan baseline lebih dulu — ia yang
menyalin GT ke `/vol/eval/gt`.

```bash
modal run --detach modal_train.py::infer --fga-mode none --num-steps 1 \
  --data-tag mix --split-by scene --val-scenes 20 --per-scene 2 --out-name baseline_mix
modal run --detach modal_train.py::infer --fga-mode partial --tag v7 --num-steps 1 \
  --data-tag mix --split-by scene --val-scenes 20 --per-scene 2
modal run --detach modal_train.py::infer --fga-mode full --tag v7 --num-steps 1 \
  --data-tag mix --split-by scene --val-scenes 20 --per-scene 2
```

```bash
modal run modal_train.py::gate_diff --tags partial_v7_s1,full_v7_s1 --baseline baseline_mix
modal run modal_train.py::sharpness --names baseline_mix,partial_v7_s1,full_v7_s1 \
  --data-tag mix --split-by scene --val-scenes 20 --per-scene 2
modal run modal_train.py::metrics --name baseline_mix
```

`gate_diff` bukan formalitas: nilai `0.0` berarti bobot FGA tidak pernah sampai
ke model, dan seluruh metrik di bawahnya hanya duplikat baseline — kegagalan
yang terbaca seperti "FGA tidak berpengaruh".

---

## Yang belum dikerjakan untuk naskah

Tiga hal yang diminta Bab 2.7 dan belum ada satu angka pun:

1. **Dataset evaluasi resmi** — ImageNet-Test, RealSRV3, RealSet80. Semua angka
   di dokumen ini dari held-out LSDIR/FFHQ. RealSet80 sudah ada di
   `testdata/RealSet80` dan bisa dievaluasi tanpa unduhan.
2. **Uji statistik** — paired t-test atau Wilcoxon per citra, koreksi
   Bonferroni/Benjamini-Hochberg, interval kepercayaan 95%.
3. **Overhead komputasi** — parameter tambahan, milidetik per citra, puncak
   memori GPU, dibandingkan antar-varian.
