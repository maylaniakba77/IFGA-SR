# Hasil Evaluasi — InvSR + FGA

Hasil utama penelitian: integrasi Fourier-Guided Attention pada decoder VAE
InvSR, dievaluasi pada dataset LSDIR+FFHQ sesuai protokol resmi InvSR.

Dokumen pendamping: [`EXPERIMENTS.md`](EXPERIMENTS.md) (riwayat seluruh konfigurasi),
[`TRAINING.md`](TRAINING.md) (desain dan alasan tiap komponen loss).

---

## Ringkasan

Pada perilaku terlatih (`gain 1.0`), FGA **mengalahkan baseline InvSR di seluruh
metrik referensi**, dengan peningkatan terbesar pada pita frekuensi tinggi —
tepat komponen yang menjadi sasaran rancangan modul.

| | partial | full |
|---|---|---|
| PSNR | **+0,60 dB** (+2,2%) | **+0,60 dB** (+2,3%) |
| SSIM | +0,0275 (+3,6%) | +0,0284 (+3,7%) |
| LPIPS-VGG | −5,4% | **−7,8%** |
| LPIPS-ALEX | −0,9% | −2,4% |
| FRC-AUC | **+11,4%** | +10,6% |
| **FRC pita tinggi** | **+20,0%** | +16,8% |
| CLIPIQA | −16,3% | −13,4% |
| MUSIQ | −3,5% | −1,2% |

Biayanya penurunan metrik no-reference — trade-off distorsi–persepsi yang klasik
dan diharapkan.

---

## Protokol evaluasi

| | |
|---|---|
| Dataset latih | 5.000 LSDIR + 5.000 FFHQ, degradasi Real-ESRGAN (protokol InvSR) |
| Split | berbasis scene, 20 scene ditahan dari training |
| Set evaluasi | 40 citra dari 20 scene, 2 undian degradasi per scene |
| Faktor SR | ×4 (LR 128 → HR 512) |
| Langkah sampling | 1 |
| Backbone | SD-Turbo + noise predictor InvSR, **seluruhnya dibekukan** |
| Parameter dilatih | modul FGA saja (~0,3 juta) |

Setiap kondisi dievaluasi pada **daftar berkas yang identik**. Ini syarat
kebenaran, bukan kenyamanan: varians statistik frekuensi antar-scene cukup besar
untuk membalik kesimpulan bila kondisi diukur pada gambar berbeda.

Verifikasi pemisahan data: nol tumpang tindih berkas **dan** nol tumpang tindih
scene antara training (80 scene) dan evaluasi (20 scene).

---

## Hasil lengkap

Acuan `baseline_onmix` = InvSR tanpa FGA, 40 citra.

### gain 1.0 — perilaku terlatih

| metrik | baseline | partial | | full | |
|---|---|---|---|---|---|
| PSNR ↑ | 26,45 | 27,04 · +2,2% | **baik** | **27,05** · +2,3% | **baik** |
| SSIM ↑ | 0,7575 | 0,7850 · +3,6% | **baik** | **0,7859** · +3,7% | **baik** |
| LPIPS-VGG ↓ | 0,3400 | 0,3216 · −5,4% | **baik** | **0,3136** · −7,8% | **baik** |
| LPIPS-ALEX ↓ | 0,2072 | 0,2054 · −0,9% | **baik** | **0,2022** · −2,4% | **baik** |
| FRC-AUC ↑ | 0,4371 | **0,4870** · +11,4% | **baik** | 0,4836 · +10,6% | **baik** |
| FRC tinggi ↑ | 0,3829 | **0,4593** · +20,0% | **baik** | 0,4473 · +16,8% | **baik** |
| CLIPIQA ↑ | **0,6686** | 0,5598 · −16,3% | buruk | 0,5789 · −13,4% | buruk |
| MUSIQ ↑ | **73,31** | 70,77 · −3,5% | buruk | 72,44 · −1,2% | buruk |

**6 baik, 2 buruk** untuk kedua mode.

### gain −0.5 — residual dibalik

| metrik | baseline | partial | | full | |
|---|---|---|---|---|---|
| PSNR ↑ | **26,45** | 25,83 · −2,3% | buruk | 26,07 · −1,4% | buruk |
| SSIM ↑ | **0,7575** | 0,7331 · −3,2% | buruk | 0,7381 · −2,6% | buruk |
| LPIPS-VGG ↓ | **0,3400** | 0,3579 · +5,3% | buruk | 0,3628 · +6,7% | buruk |
| LPIPS-ALEX ↓ | **0,2072** | 0,2223 · +7,3% | buruk | 0,2261 · +9,1% | buruk |
| FRC-AUC ↑ | **0,4371** | 0,4081 · −6,6% | buruk | 0,4207 · −3,8% | buruk |
| FRC tinggi ↑ | **0,3829** | 0,3403 · −11,1% | buruk | 0,3735 · −2,5% | buruk |
| CLIPIQA ↑ | 0,6686 | **0,6907** · +3,3% | **baik** | 0,6832 · +2,2% | **baik** |
| MUSIQ ↑ | 73,31 | **73,44** · +0,2% | baik¹ | 71,87 · −2,0% | buruk |

¹ +0,2% berada di bawah ambang bermakna (~1,0 untuk MUSIQ); tidak layak diklaim.

> **Tanda persentase bukan penilaian.** Pada LPIPS, nilai yang turun berarti
> membaik — ia mengukur jarak ke ground truth. Kolom penilaian dipisah karena itu.

---

## Trade-off distorsi–persepsi terhadap `gain`

Diurutkan menurut gain, seluruh metrik bergerak **monoton**:

| gain | PSNR | SSIM | LPIPS-VGG | FRC-AUC | FRC tinggi | CLIPIQA |
|---|---|---|---|---|---|---|
| −0,5 (partial) | 25,83 | 0,7331 | 0,3579 | 0,4081 | 0,3403 | **0,6907** |
| −0,5 (full) | 26,07 | 0,7381 | 0,3628 | 0,4207 | 0,3735 | 0,6832 |
| **0 (baseline)** | 26,45 | 0,7575 | 0,3400 | 0,4371 | 0,3829 | 0,6686 |
| 1,0 (partial) | 27,04 | 0,7850 | 0,3216 | **0,4870** | **0,4593** | 0,5598 |
| 1,0 (full) | **27,05** | **0,7859** | **0,3136** | 0,4836 | 0,4473 | 0,5789 |

Naik seiring gain: PSNR, SSIM, LPIPS-VGG, FRC-AUC, FRC tinggi.
Turun seiring gain: CLIPIQA.

`gain` adalah pengali cabang residual FGA saat inferensi — `output = base +
gain × delta`. Nilai 0 melewati FGA sepenuhnya dan menghasilkan baseline persis;
nilai negatif membalik koreksi menjadi unsharp mask.

Karena ia parameter inferensi, bukan bobot terlatih, **seluruh kurva di atas
dihasilkan satu checkpoint tanpa pelatihan ulang**. Ini memberi kendali pasca-
training atas posisi di kurva trade-off, dan merupakan kontribusi tambahan di
luar perbaikan rekonstruksi itu sendiri.

---

## Metrik FRC

Fourier Ring Correlation mengukur korelasi antara hasil SR dan ground truth
**secara terpisah untuk tiap cincin frekuensi**, lalu memberi bobot sama pada
tiap cincin:

```
FRC(q) = Re[ Σ F_sr · conj(F_gt) ] / sqrt( Σ|F_sr|² · Σ|F_gt|² )
```

Mengapa ini penting di sini: PSNR dan SSIM didominasi komponen frekuensi rendah,
karena spektrum citra natural meluruh kira-kira 1/f. Kesalahan pada detail halus
nyaris tidak memengaruhi keduanya. FRC tidak punya bias itu.

| pita | rentang | ukuran fitur |
|---|---|---|
| rendah | 0–33% Nyquist | > 6 piksel — bentuk besar, gradien |
| menengah | 33–67% | 3–6 piksel — tekstur sedang |
| **tinggi** | 67–100% | 2–3 piksel — **detail terhalus** |

Rincian per pita pada gain 1.0:

| pita | baseline | partial | full |
|---|---|---|---|
| rendah | 0,6523 | 0,6732 (+3,2%) | 0,6761 (+3,6%) |
| menengah | 0,2766 | 0,3286 (+18,8%) | 0,3279 (+18,5%) |
| **tinggi** | 0,3829 | **0,4593 (+20,0%)** | 0,4473 (+16,8%) |

Keunggulannya **terkonsentrasi di pita menengah dan tinggi** — pita rendah hanya
naik 3%. Ini bukti mekanistik bahwa modulnya bekerja sesuai rancangan, bukan
peningkatan menyeluruh yang kebetulan.

Sebagai pembanding, FGA-SR asli melaporkan peningkatan konsistensi domain-
frekuensi hingga 29% pada lima backbone SR regresi. Hasil di sini berada pada
orde yang sama, pada backbone difusi laten yang belum pernah diuji sebelumnya.

---

## Perbandingan `partial` vs `full` (H3)

| | partial unggul | full unggul |
|---|---|---|
| gain 1.0 | FRC-AUC (+0,003), FRC tinggi (+0,012) | PSNR (+0,01), SSIM (+0,001), LPIPS-VGG (+0,008), LPIPS-ALEX (+0,003) |

Seluruh selisih berada **di bawah ambang bermakna** untuk metrik masing-masing.

`partial` memasang FGA hanya pada blok upsampling terakhir; `full` pada
ketiganya, sehingga jumlah parameternya sekitar tiga kali lipat. Karena kualitas
keduanya praktis setara, **koreksi tahap terakhir sudah menangkap hampir seluruh
keuntungan** dan injeksi multi-skala tidak sepadan dengan biayanya.

---

## Batasan

**Belum ada uji statistik.** Naskah Bab 2.7 menjanjikan paired t-test atau
Wilcoxon per citra dengan koreksi Bonferroni/Benjamini-Hochberg dan interval
kepercayaan 95%. Tanpa itu, selisih 0,60 dB belum dapat dibedakan secara formal
dari variasi antar-citra.

**Belum ada dataset evaluasi resmi.** Angka di sini dari held-out LSDIR/FFHQ,
bukan ImageNet-Test, RealSRV3, atau RealSet80 yang ditetapkan naskah. RealSet80
sudah tersedia di `testdata/RealSet80` dan dapat dievaluasi tanpa unduhan.

**Hanya satu rezim sampling.** Seluruh hasil pada `num_steps = 1`. Matriks
eksperimen naskah menuntut 1, 3, dan 5 langkah.

**Overhead komputasi belum diukur.** Jumlah parameter tambahan, waktu inferensi
per citra, dan puncak memori GPU belum dibandingkan antar-varian — padahal itu
argumen kunci naskah bahwa kecepatan InvSR tidak boleh dikorbankan.

**40 citra dari 20 scene.** Cukup untuk perbandingan internal yang konsisten,
tetapi set yang lebih besar akan memperkuat generalisasi.
