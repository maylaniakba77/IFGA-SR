"""
losses.py — Fungsi kerugian gabungan untuk fine-tuning modul FGA
pada blok upsampling decoder VAE InvSR.

    L_total = w_pixel * L_pixel + w_freq * L_frequency [+ w_lpips * L_LPIPS]

- L_pixel     : L1 (default) atau L2 pada ruang piksel terhadap ground-truth HR.
- L_frequency : L1 pada domain frekuensi (magnitudo selisih FFT-2D kompleks),
                mengikuti rumusan frequency-domain L1 loss pada FGA-SR.
- L_LPIPS     : opsional (default nonaktif), metrik persepsi berbasis deep feature.

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
        freq_mode  : 'full' | 'highpass'
        freq_cutoff: cutoff untuk freq_mode='highpass'
        lpips_net  : backbone LPIPS ('alex' lebih ringan, 'vgg' lebih standar)
    """

    def __init__(
        self,
        pixel_type: str = "l1",
        w_pixel: float = 1.0,
        w_freq: float = 0.1,
        w_lpips: float = 0.0,
        freq_mode: str = "full",
        freq_cutoff: float = 0.25,
        lpips_net: str = "alex",
    ):
        super().__init__()
        assert pixel_type in ("l1", "l2"), f"pixel_type tidak dikenal: {pixel_type}"
        assert freq_mode in ("full", "highpass"), f"freq_mode tidak dikenal: {freq_mode}"

        self.pixel_type = pixel_type
        self.w_pixel = w_pixel
        self.w_freq = w_freq
        self.w_lpips = w_lpips
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
        else:
            l_freq = frequency_l1_loss_highpass(pred, target, cutoff=self.freq_cutoff)

        total = self.w_pixel * l_pix + self.w_freq * l_freq
        parts = {
            "loss_pixel": float(l_pix.detach()),
            "loss_freq": float(l_freq.detach()),
        }

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
