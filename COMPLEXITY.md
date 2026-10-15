# Kompleksitas Komputasi — InvSR + FGA

Analisis asimtotik (Big-O) waktu dan ruang untuk inferensi InvSR, pengaruh
penyisipan FGA, dan biaya training FGA. Semua klaim diturunkan dari kode di
repo ini; baris sumbernya dirujuk di setiap bagian.

## Ringkasan

| | Waktu | Ruang |
|---|---|---|
| InvSR (dengan chopping, default repo) | **O(N · n)** | **O(n)** |
| InvSR + FGA (`partial` / `full`) | **O(N · n)** | **O(n)** |
| Tambahan dari FGA | O(n) | O(n) |
| InvSR tanpa chopping (teoretis) | O(N · n²) | O(n²) atau O(n)* |
| Satu iterasi training FGA | O(n_crop) | O(n_crop) |

\* O(n) bila kernel attention-nya flash/memory-efficient, yang tidak
menyimpan matriks n × n.

Tiga kesimpulan utama:

1. **FGA tidak mengubah orde kompleksitas InvSR**, baik waktu maupun ruang.
2. **Overhead relatif FGA berbanding O(1/N)**, sehingga porsinya mengecil
   seiring bertambahnya langkah sampling. Overhead terbesar ada pada rezim
   1 langkah.
3. **Ruang tidak bergantung pada N.** Latent ditimpa setiap langkah, jadi
   1 dan 5 langkah memakai VRAM puncak yang sama.

---

## Notasi

| Simbol | Arti |
|---|---|
| n = h · w | jumlah piksel citra LR |
| 16n | jumlah piksel citra HR (SR ×4) |
| n/4 | jumlah token latent (HR/8 per sisi) |
| N | jumlah langkah sampling (1, 3, atau 5) |
| L = 77 | panjang token prompt (tetap) |
| c, d | lebar kanal backbone dan lebar internal FGA (`inner_dim = 64`) |
| p = 128 | sisi patch LR untuk chopping |
| M, I | jumlah sampel training dan jumlah iterasi |

Konstanta arsitektur (c, d, L, p) diperlakukan sebagai O(1). Variabelnya hanya
n dan N.

---

## 1. Kompleksitas waktu

### 1.1 Per komponen, tanpa chopping

| Komponen | Dipanggil | Bagian linear | Bagian kuadratik (self-attention) | Total |
|---|---|---|---|---|
| Text encoder | 1× | — | L² (konstan) | O(1) |
| Noise predictor | 1× | konvolusi O(n) | attention pada n token LR → O(n²) | O(n²) |
| VAE encoder | 1× | konvolusi di HR, O(16n) | mid-block, n/4 token → O(n²/16) | O(n²) |
| UNet | **N×** | konvolusi + cross-attention (n/4 × L) → O(n) | self-attention, n/4 token → O(n²/16) | **O(N · n²)** |
| VAE decoder | 1× | konvolusi di HR, O(16n) | mid-block, n/4 token → O(n²/16) | O(n²) |
| FGA | 1× (di dalam decoder) | O(n · (c·d + d²)) | tidak ada | O(n) |

Totalnya:

```
T(n, N) = O(n²) + O(N · n²) + O(N · n)  =  O(N · n²)
```

Attention dengan token terbanyak ada di **noise predictor**, bukan di UNet.
Noise predictor adalah `TimeAwareEncoder` dengan `conv_in` stride 1
(`src/diffusers/models/autoencoders/vae.py:236`), sehingga `AttnDownBlock2D`
pertamanya bekerja pada n token LR. UNet hanya melihat n/4 token latent. Bedanya,
noise predictor dipanggil sekali, sedangkan UNet dipanggil N kali.

### 1.2 Dengan chopping (perilaku sebenarnya di repo ini)

Input yang lebih besar dari 128 LR dipotong menjadi patch p × p dengan stride
p/2 = 64 (`sampler_invsr.py:201-204`). Setiap patch berbiaya tetap:

```
T_patch(N) = A + N · B        (A, B konstan terhadap n)
```

Jumlah patch-nya:

```
P ≈ n / 64²        (setiap piksel tertutup ±4 patch karena tumpang-tindih 50%)
```

Sehingga:

```
T(n, N) = P · (A + N · B)  =  O(N · n)
```

Biaya bergantung pada ukuran input dalam tiga rezim:

| Ukuran LR | Perilaku | Orde terhadap n |
|---|---|---|
| ≤ 128 | di-*pad* menjadi 128, satu patch | O(1) |
| > 128 | chopping, jumlah patch ∝ n | O(n) |
| tanpa chopping | attention global | O(n²) |

Text encoder dijalankan sekali per batch patch (`extra_bs = 8`), jadi ia juga
tumbuh linear terhadap n, dengan konstanta yang sangat kecil.

### 1.3 Terhadap jumlah langkah N

```
T(N) = A + N · B
A = text encoder + noise predictor + VAE encoder + VAE decoder   (sekali)
B = satu forward UNet                                           (diulang N kali)
```

Biaya ini linear terhadap N, karena hanya UNet yang diulang.
`intermediate_noise_predictor` tidak dipasang (`sampler_invsr.py:108` hanya
mengisi `start_noise_predictor`), jadi tidak ada komponen lain yang ikut
terulang.

Jadwal timestep per N (`inference_invsr.py:67-76`):

| N | Timestep |
|---|---|
| 1 | [200] |
| 3 | [200, 100, 50] |
| 5 | [250, 200, 150, 100, 50] |

Orde O(N · n) sama dengan SR berbasis difusi lainnya. Kontribusi InvSR terletak
pada N yang kecil (1–5), bukan 15–200 seperti metode multi-langkah.

---

## 2. Kompleksitas ruang

| Bagian memori | Orde | Alasan |
|---|---|---|
| Bobot model (±1,3 miliar parameter) | O(1) | ukurannya tetap, tidak bergantung pada gambar |
| Aktivasi satu patch | O(1) | patch selalu 128 LR / 512 HR |
| Latent antar-langkah | O(1) per patch | ditimpa setiap langkah, tidak disimpan |
| Buffer keluaran + peta bobot penggabung patch | **O(n)** | menampung seluruh citra HR (16n piksel) |
| **Total** | **O(n)** | |

### Mengapa N tidak muncul

Loop sampling hanya menyimpan satu latent pada satu waktu
(`src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inversion_sr.py:1115`):

```python
latents = self.scheduler.step(noise_pred, t, latents, ...)[0]   # menimpa
```

Inferensi juga berjalan di bawah `torch.no_grad()`, sehingga tidak ada aktivasi
yang disimpan untuk backward. Waktu bertambah seiring N, tetapi ruang tidak.

### Tanpa chopping

| Kernel attention | Ruang |
|---|---|
| attention biasa (matriks n × n disimpan) | O(n²) |
| flash / memory-efficient attention | O(n) |

---

## 3. Kompleksitas FGA

### 3.1 Waktu

Konfigurasi di `fga_integration/fga_upsampler.py`: `window_size = 1`,
`overlap_ratio = 4`, `upscale = 2`. Dari `fga/archs/fga_arch.py:72-77`:

```
overlap_win_size = 1 · 4 + 1 = 5   →  25 key per posisi LR
window_size_up   = 2 · 1     = 2   →   4 query per posisi LR
```

Jendela attention-nya tetap dan lokal, jadi biayanya linear. Untuk satu
upsampler dengan input berukuran h′ × w′ dan c kanal:

| Sub-operasi | Biaya |
|---|---|
| embed conv 3×3 | 9 · c · d · h′w′ |
| SubPixelMLP (conv d → 4d) | 36 · d² · h′w′ |
| proyeksi K, V | 2 · d² · h′w′ |
| attention (QKᵀ dan AV) | 2 · 4 · 25 · d · h′w′ = 200 · d · h′w′ |
| proyeksi + MLP di HR | ≈ 2 · d² · 4h′w′ |
| unembed conv 3×3 di HR | 9 · d · c · 4h′w′ |
| **Total** | **O(h′w′ · (c·d + d²)) = O(n)** |

### 3.2 `partial` vs `full`

| Mode | Upsampler | Posisi input h′w′ | Orde |
|---|---|---|---|
| `partial` | terakhir saja | 4n | O(n) |
| `full` | ketiganya | n/4 + n + 4n = 5,25n | O(n) |

Dalam jumlah posisi, `full` ≤ 1,31 × `partial` (deret geometris). Ordenya sama,
hanya konstantanya yang berbeda. Faktor kanal c juga berbeda antar-upsampler
(512 di dua yang pertama, 256 di yang terakhir), sehingga rasio FLOPs
sebenarnya perlu diukur, bukan diturunkan dari jumlah posisi saja.

### 3.3 Overhead terhadap InvSR

```
T_InvSR+FGA(n, N) = O(N · n) + O(n) = O(N · n)

overhead relatif  = C_FGA / (A + N · B)  =  O(1/N)
```

Porsi biaya FGA paling besar pada 1 langkah. Pada 3 langkah porsinya kira-kira
sepertiga dari itu, karena hanya penyebutnya (biaya UNet) yang bertambah.

### 3.4 Ruang

FGA menambah O(n), tetapi konstantanya cukup besar. `nn.Unfold`
(`fga/archs/fga_arch.py:75`) membentuk tensor K/V berukuran 2 · d · 25 nilai
per posisi. Di upsampler terakhir ada 4n posisi, dan semuanya dalam float32
karena FGA sengaja dijalankan di presisi penuh (`inject_fga` memanggil
`new.fga.float()`). Karena itu tambahan VRAM dari FGA bisa terasa walaupun
ordenya tidak berubah.

---

## 4. Kompleksitas training FGA

Backbone dibekukan, jadi latent dihitung sekali (`cache`) lalu dipakai ulang di
setiap iterasi.

| Tahap | Waktu | Ruang | Keterangan |
|---|---|---|---|
| `cache` | O(M · N · n) | O(M · n), di disk | sekali; jalankan sampling penuh per sampel |
| satu iterasi `train` | O(n_crop) | O(n_crop), di GPU | hanya forward + backward decoder |
| total training | O(M · N · n + I · n_crop) | | |

Konsekuensinya:

- **Biaya iterasi training tidak bergantung pada N.** `cache --num-steps 3`
  kira-kira 3× lebih lama daripada `--num-steps 1`, tetapi durasi training-nya
  sama.
- **Ruang training juga tidak bergantung pada N**, karena yang di-backprop
  hanya decoder.

---

## 5. Verifikasi empiris

Analisis di atas bisa dicek dengan `scripts/complexity.py` (lihat
`MODAL.md` §10c):

```bash
modal run modal_train.py::complexity --lr-sizes 128,256,512 --out-name complexity_scaling
```

Prediksi yang bisa diuji:

| Prediksi | Yang diharapkan pada keluaran |
|---|---|
| Waktu linear terhadap n | jumlah patch LR 128/256/512 = 1/9/49, jadi GFLOPs berbanding ≈ 1 : 9 : 49. Untuk gambar sangat besar rasionya mendekati 4× setiap kali sisi digandakan |
| Waktu linear terhadap N | GFLOPs(N) − GFLOPs(1) = (N − 1) × GFLOPs UNet |
| Ruang tidak bergantung pada N | VRAM puncak sama untuk 1, 3, dan 5 langkah |
| FGA tidak mengubah orde | Δ GFLOPs FGA konstan terhadap N; persentasenya turun seiring N |
| Overhead `full` ≈ `partial` | Δ `full` hanya sedikit di atas Δ `partial` |

Plot log-log biaya terhadap n juga berguna: kemiringan ≈ 1 berarti linear,
≈ 2 berarti kuadratik.

Laporkan nama GPU bersama angka latensi. Latensi dari GPU berbeda tidak
sebanding, sedangkan FLOPs dan orde kompleksitasnya tidak bergantung pada
perangkat keras.
