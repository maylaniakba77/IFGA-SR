"""
losses.py — Fungsi kerugian gabungan untuk fine-tuning modul FGA
pada blok upsampling decoder VAE InvSR.

    L_total = w_pixel * L_pixel + w_freq * L_frequency [+ w_lpips * L_LPIPS]

- L_pixel     : L1 (default) atau L2 pada ruang piksel terhadap ground-truth HR.
- L_frequency : bergantung `freq_mode`:
                  'full'      L1 pada selisih FFT-2D KOMPLEKS (seluruh spektrum)
                  'highpass'  sama, tetapi dibatasi band frekuensi tinggi
                  'magnitude' L1 pada selisih MAGNITUDO spektrum band tinggi
- L_LPIPS     : opsional (default nonaktif), metrik persepsi berbasis deep feature.

PERINGATAN PENTING SOAL 'full' DAN 'highpass'
    Keduanya menghitung selisih bilangan KOMPLEKS pada basis `norm="ortho"`.
    rFFT ortonormal adalah transformasi uniter, sehingga (teorema Parseval) jarak
    di domain frekuensi ekuivalen dengan jarak di domain piksel — term ini BUKAN
    sinyal supervisi baru, hanya loss fidelitas yang sama dalam basis terotasi.
    Konsekuensinya ia SENSITIF FASE: tekstur yang benar secara statistik tetapi
    bergeser beberapa piksel dihukum berat, persis seperti L1 piksel. Arah
    gradiennya karena itu "hilangkan detail yang tidak sejajar", bukan "tambahkan
    detail" — hasil pelatihan menjadi lebih halus dari baseline.

    'magnitude' mencocokkan ENERGI spektrum tanpa mewajibkan kecocokan fase, jadi
    arah gradiennya "samakan kekayaan detail". Inilah varian yang dipakai bila
    tujuannya ketajaman. Pertahankan 'full'/'highpass' hanya sebagai kondisi
    pembanding pada studi ablasi.

CATATAN RENTANG NILAI
    `pred` dan `target` HARUS berada pada rentang yang sama.
    Keluaran `vae.decode(...).sample` berada pada [-1, 1], jadi ground-truth HR
    juga dinormalisasi ke [-1, 1] pada train_fga.py. LPIPS juga mengharapkan
    [-1, 1], sehingga konsisten.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Frequency-domain loss
# --------------------------------------------------------------------------- #
def frequency_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L1 pada domain frekuensi.

    Menghitung |F(pred) - F(target)| dengan F = FFT-2D real (rfft2), sehingga
    error magnitudo DAN fase ikut terukur. Ini memberi sinyal supervisi langsung
    pada spektrum Fourier — mengatasi bias loss piksel yang secara statistik
    didominasi komponen frekuensi rendah.

    FFT dipaksa ke float32: torch.fft tidak stabil (dan pada beberapa versi tidak
    didukung) untuk float16.
    """
    pred_f = torch.fft.rfft2(pred.float(), norm="ortho")
    tgt_f = torch.fft.rfft2(target.float(), norm="ortho")
    diff = pred_f - tgt_f
    # sqrt(re^2 + im^2) = magnitudo bilangan kompleks; eps menjaga gradien di 0
    mag = torch.sqrt(diff.real.pow(2) + diff.imag.pow(2) + eps)
    return mag.mean()


def frequency_l1_loss_highpass(
    pred: torch.Tensor,
    target: torch.Tensor,
    cutoff: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Varian opsional: hanya menghitung error pada komponen frekuensi TINGGI.

    Berguna untuk studi ablasi — menguji apakah menekankan band frekuensi tinggi
    secara eksplisit memberi manfaat tambahan dibanding L1 frekuensi penuh.

    Args:
        cutoff: fraksi frekuensi Nyquist yang dianggap "rendah" dan dibuang
                (0.25 = buang 25% frekuensi terendah pada tiap sumbu).
    """
    pred_f = torch.fft.rfft2(pred.float(), norm="ortho")
    tgt_f = torch.fft.rfft2(target.float(), norm="ortho")

    _, _, h, w = pred_f.shape
    fy = torch.fft.fftfreq(pred.shape[-2], device=pred.device).abs()
    fx = torch.fft.rfftfreq(pred.shape[-1], device=pred.device).abs()
    mask = ((fy[:, None] > cutoff * 0.5) | (fx[None, :] > cutoff * 0.5)).float()
    mask = mask[None, None]  # (1, 1, H, W//2+1)

    diff = (pred_f - tgt_f) * mask
    mag = torch.sqrt(diff.real.pow(2) + diff.imag.pow(2) + eps)
    denom = mask.sum().clamp(min=1.0) * pred.shape[0] * pred.shape[1]
    return mag.sum() / denom


def spectral_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cutoff: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L1 pada selisih MAGNITUDO spektrum, dibatasi band frekuensi tinggi.

    Berbeda dari `frequency_l1_loss*`, di sini fase dibuang lebih dulu lewat
    `.abs()`. Yang disupervisi adalah seberapa besar energi yang dimiliki tiap
    frekuensi, bukan di mana persisnya detail itu berada.

    Mengapa ini yang mendorong ketajaman: latent InvSR bersifat generatif, jadi
    detail frekuensi tingginya plausibel tetapi tidak sejajar piksel dengan GT.
    Loss yang sensitif fase hanya bisa menurunkan error dengan MEREDAM detail
    tersebut — meredam selalu aman baginya. Loss magnitudo tidak: begitu energi
    band tinggi turun di bawah GT, loss NAIK. Jadi penghalusan tidak lagi
    menjadi jalan keluar yang murah.

    Perhatikan bahwa loss ini SIMETRIS — energi yang berlebih dihukum sama
    beratnya dengan yang kurang. Itu memang disengaja: ia menjaga spektrum
    keluaran tetap menyerupai citra natural, bukan mendorongnya menuju noise.

    Args:
        cutoff: batas bawah band yang disupervisi, sebagai fraksi frekuensi
                Nyquist (0.25 = hanya frekuensi di atas 25% Nyquist).
    """
    pred_f = torch.fft.rfft2(pred.float(), norm="ortho")
    tgt_f = torch.fft.rfft2(target.float(), norm="ortho")

    fy = torch.fft.fftfreq(pred.shape[-2], device=pred.device).abs()
    fx = torch.fft.rfftfreq(pred.shape[-1], device=pred.device).abs()
    # cutoff * 0.5: fftfreq bersatuan cycles/sample pada [0, 0.5], sehingga
    # Nyquist = 0.5. Konvensi ini identik dengan frequency_l1_loss_highpass,
    # karena keduanya dibaca dari flag --freq_cutoff yang sama.
    mask = ((fy[:, None] > cutoff * 0.5) | (fx[None, :] > cutoff * 0.5)).float()[None, None]

    # .abs() pada spektrum -> magnitudo; selisihnya tidak lagi bergantung fase.
    # eps menjaga gradien sqrt di dalam abs() tetap terdefinisi pada nol.
    diff = (torch.sqrt(pred_f.real.pow(2) + pred_f.imag.pow(2) + eps)
            - torch.sqrt(tgt_f.real.pow(2) + tgt_f.imag.pow(2) + eps)).abs() * mask
    denom = mask.sum().clamp(min=1.0) * pred.shape[0] * pred.shape[1]
    return diff.sum() / denom


# --------------------------------------------------------------------------- #
# Sharpness loss — satu arah, target dapat melampaui GT
# --------------------------------------------------------------------------- #
_LAP_TRAIN = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def _lap_energy(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Energi Laplacian per citra (bisa dibackprop). x: (B,3,H,W) di [-1,1]."""
    lum = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    lap = F.conv2d(lum, _LAP_TRAIN.to(lum.device, lum.dtype))
    return lap[..., 2:-2, 2:-2].flatten(1).pow(2).mean(dim=1) + eps


def sharpness_deficit_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ratio: float = 1.0,
) -> torch.Tensor:
    """Hukum HANYA bila `pred` KURANG tajam dari `ratio` x ketajaman GT.

    L = relu(ratio - energi_laplacian(pred) / energi_laplacian(gt))

    Tiga sifat yang membuatnya berbeda dari setiap loss lain di berkas ini:

    1. SATU ARAH. Terlalu tajam sama sekali tidak dihukum, jadi tidak ada
       tekanan balik ke arah blur. Setiap loss sebelumnya simetris atau berupa
       jarak, sehingga penghalusan selalu menjadi jalan keluar murah.

    2. TIDAK BISA DIPUASKAN DENGAN MENGHALUSKAN. Menghaluskan menurunkan energi
       Laplacian, yang MENAIKKAN loss. Arah gradiennya secara struktural adalah
       "tambah kontras lokal".

    3. TARGETNYA DAPAT MELAMPAUI GT. `ratio` adalah pilihanmu, bukan properti
       data. ratio=1.0 menyamai foto asli; ratio=1.6 menyamai ketajaman baseline
       InvSR pada set 4-scene; ratio=2.0 melampauinya. Inilah satu-satunya
       parameter di repo ini yang dapat meminta keluaran LEBIH tajam dari
       baseline.

    Buta fase: yang dibandingkan statistik energi, bukan posisi piksel. Tekstur
    yang bergeser tidak dihukum.

    PERINGATAN: mengejar ratio tinggi akan menaikkan ketajaman terukur, tetapi
    kontras lokal dapat dinaikkan dengan noise maupun dengan detail. Selalu
    dampingi dengan `--w_gan` > 0, yang menuntut teksturnya terlihat nyata,
    dan verifikasi dengan CLIPIQA/MUSIQ.
    """
    return F.relu(ratio - _lap_energy(pred) / _lap_energy(target)).mean()


# --------------------------------------------------------------------------- #
# Combined loss module
# --------------------------------------------------------------------------- #
class FGALoss(nn.Module):
    """Kerugian gabungan untuk fine-tuning modul FGA.

    Args:
        pixel_type : 'l1' | 'l2'
        w_pixel    : bobot komponen piksel
        w_freq     : bobot komponen domain-frekuensi
        w_lpips    : bobot LPIPS (0.0 = nonaktif, tidak memuat model LPIPS)
        freq_mode  : 'full' | 'highpass' | 'magnitude'
        freq_cutoff: cutoff untuk freq_mode='highpass'
        lpips_net  : backbone LPIPS ('alex' lebih ringan, 'vgg' lebih standar)
    """

    def __init__(
        self,
        pixel_type: str = "l1",
        w_pixel: float = 1.0,
        w_freq: float = 0.1,
        w_lpips: float = 0.0,
        w_sharp: float = 0.0,
        sharp_ratio: float = 1.0,
        w_range: float = 1.0,
        freq_mode: str = "full",
        freq_cutoff: float = 0.25,
        lpips_net: str = "alex",
    ):
        super().__init__()
        assert pixel_type in ("l1", "l2"), f"pixel_type tidak dikenal: {pixel_type}"
        assert freq_mode in ("full", "highpass", "magnitude"), \
            f"freq_mode tidak dikenal: {freq_mode}"

        self.pixel_type = pixel_type
        self.w_pixel = w_pixel
        self.w_freq = w_freq
        self.w_lpips = w_lpips
        self.w_sharp = w_sharp
        self.sharp_ratio = sharp_ratio
        self.w_range = w_range
        self.freq_mode = freq_mode
        self.freq_cutoff = freq_cutoff

        self.lpips = None
        if w_lpips > 0:
            try:
                import lpips as _lpips  # pip install lpips
            except ImportError as e:
                raise ImportError(
                    "w_lpips > 0 memerlukan paket `lpips`. Jalankan: pip install lpips"
                ) from e
            self.lpips = _lpips.LPIPS(net=lpips_net)
            for p in self.lpips.parameters():
                p.requires_grad_(False)
            self.lpips.eval()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred, target: (B, 3, H, W) pada rentang [-1, 1]

        Returns:
            total  : skalar tensor untuk backward()
            parts  : dict komponen loss (float) untuk logging
        """
        if self.pixel_type == "l1":
            l_pix = F.l1_loss(pred, target)
        else:
            l_pix = F.mse_loss(pred, target)

        if self.freq_mode == "full":
            l_freq = frequency_l1_loss(pred, target)
        elif self.freq_mode == "highpass":
            l_freq = frequency_l1_loss_highpass(pred, target, cutoff=self.freq_cutoff)
        else:
            l_freq = spectral_magnitude_loss(pred, target, cutoff=self.freq_cutoff)

        total = self.w_pixel * l_pix + self.w_freq * l_freq
        parts = {
            "loss_pixel": float(l_pix.detach()),
            "loss_freq": float(l_freq.detach()),
        }

        if self.w_sharp > 0:
            l_sharp = sharpness_deficit_loss(pred, target, self.sharp_ratio)
            total = total + self.w_sharp * l_sharp
            parts["loss_sharp"] = float(l_sharp.detach())

        if self.w_range > 0:
            # Hukum nilai piksel di luar [-1, 1].
            #
            # WAJIB bila w_sharp > 0. `sharpness_deficit_loss` satu arah dan tak
            # berbatas, dan dihitung pada keluaran yang BELUM di-clamp — sehingga
            # optimizer bisa mendapat energi Laplacian secara gratis dengan
            # mendorong piksel jauh ke luar rentang. Saat inferensi nilai itu
            # terpotong per-kanal dan muncul sebagai bintik magenta terang
            # (teramati: 273 piksel [231, 58, 137] pada 0801_d0.png).
            #
            # Meng-clamp pred sebagai gantinya TIDAK menyelesaikannya: gradien di
            # luar rentang menjadi nol, jadi lonjakannya tidak pernah dihukum.
            l_rng = F.relu(pred.abs() - 1.0).mean()
            total = total + self.w_range * l_rng
            parts["loss_range"] = float(l_rng.detach())

        if self.lpips is not None:
            l_lpips = self.lpips(pred, target).mean()
            total = total + self.w_lpips * l_lpips
            parts["loss_lpips"] = float(l_lpips.detach())

        parts["loss_total"] = float(total.detach())
        return total, parts


# --------------------------------------------------------------------------- #
# Metrik pemantauan (bukan untuk backward)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    """PSNR untuk pemantauan selama pelatihan.

    data_range=2.0 karena tensor berada pada [-1, 1].
    Ini BUKAN pengganti evaluasi resmi di Bab II (yang dihitung pada kanal Y
    ruang YCbCr mengikuti protokol InvSR) — hanya indikator cepat saat training.
    """
    mse = F.mse_loss(pred.float(), target.float())
    if mse.item() == 0:
        return float("inf")
    return float(10.0 * torch.log10(data_range**2 / mse))


_LAP_K = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


@torch.no_grad()
def laplacian_var(x: torch.Tensor) -> float:
    """Varians Laplacian pada kanal luminansi: ukuran KONTRAS LOKAL absolut.

    Inilah metrik ketajaman yang dipakai untuk mengevaluasi pekerjaan ini, dan
    ia dipantau di sini supaya kemajuannya terlihat SELAMA training.

    Kenapa bukan fraksi energi frekuensi tinggi: fraksi HF ternyata menyesatkan.
    Baseline InvSR punya fraksi HF LEBIH TINGGI dari GT (0.5755 vs 0.5488) tetapi
    kontras tepi 29% LEBIH RENDAH — karena energi HF-nya berupa noise difus
    beramplitudo kecil, bukan tepi yang terstruktur. Varians Laplacian tidak
    tertipu oleh itu.

    Args:
        x: (B, 3, H, W) pada rentang [-1, 1].
    """
    lum = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    lap = F.conv2d(lum.float(), _LAP_K.to(lum.device, torch.float32))
    # buang 2 piksel tepi: kernel di batas tidak mencerminkan konten
    lap = lap[..., 2:-2, 2:-2]
    return float(lap.flatten(1).var(dim=1).mean())


@torch.no_grad()
def spectral_consistency(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Konsistensi domain-frekuensi (indikator ringkas).

    Didefinisikan sebagai 1 - (error spektrum ternormalisasi); makin mendekati 1
    makin konsisten spektrum keluaran terhadap ground truth. Dipakai sebagai
    proksi cepat untuk metrik pembeda utama tesis selama pelatihan.
    """
    pf = torch.fft.rfft2(pred.float(), norm="ortho").abs()
    tf = torch.fft.rfft2(target.float(), norm="ortho").abs()
    err = (pf - tf).abs().sum()
    denom = tf.abs().sum().clamp(min=eps)
    return float(1.0 - err / denom)
