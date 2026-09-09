"""
discriminator.py — PatchGAN untuk melatih FGA dengan term adversarial.

MENGAPA INI DIBUTUHKAN
    Seluruh loss sebelumnya (L1, LPIPS, magnitudo spektrum) adalah JARAK KE GT.
    Untuk SR x4 dari degradasi berat, posterior-nya lebar: banyak citra HR yang
    plausibel untuk satu LR. Peminimal jarak adalah RATA-RATA posterior, dan
    rata-rata posterior itu blur secara definisi.

    Baseline InvSR tidak meminimalkan jarak apa pun — ia mengambil SAMPEL dari
    posterior lewat difusi, dan sampel selalu lebih tajam dari rata-rata.

    Sweep gain membuktikan trade-off ini tidak bisa dihindari dengan menskala
    cabang residual: setiap metrik referensi membaik monoton menuju gain 1,
    setiap metrik no-reference memburuk monoton. Menskala hanya menggeser posisi
    di sepanjang kurva.

    Discriminator mengubah pertanyaannya. Ia tidak bertanya "seberapa dekat
    piksel ini ke GT", melainkan "apakah tekstur ini terlihat seperti tekstur
    citra nyata". Tekstur yang benar tetapi bergeser posisi TIDAK dihukum, dan
    keluaran blur dihukum karena patch blur mudah dikenali sebagai palsu.

PILIHAN DESAIN
    Spectral normalization, bukan BatchNorm. Ia membatasi konstanta Lipschitz
    discriminator sehingga training jauh lebih stabil — penting karena run ini
    berjalan detached di Modal tanpa ada yang mengawasi, dan tidak ada resume.

    Hinge loss, bukan BCE. Gradiennya tidak jenuh saat discriminator menang,
    yang merupakan mode kegagalan paling umum pada GAN untuk super-resolution.

    Tanpa kondisi (unconditional). Discriminator hanya melihat patch keluaran,
    tidak dipasangkan dengan LR — sama seperti Real-ESRGAN. Ini menghindari
    keharusan meneruskan LR ke dalam loop training yang sudah bekerja di ruang
    latent yang di-cache.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """PatchGAN 70x70 dengan spectral norm.

    Keluarannya peta skor, bukan skalar tunggal: setiap posisi menilai satu
    patch. Untuk tekstur ini lebih baik daripada discriminator global, karena
    sinyalnya lokal — persis skala tempat detail hilang.
    """

    def __init__(self, in_ch: int = 3, base: int = 64, n_layers: int = 3):
        super().__init__()
        def conv(i, o, k, s, p):
            return spectral_norm(nn.Conv2d(i, o, k, s, p))

        layers = [conv(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        ch = base
        for n in range(1, n_layers):
            nxt = min(base * 2 ** n, 512)
            layers += [conv(ch, nxt, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            ch = nxt
        nxt = min(ch * 2, 512)
        layers += [conv(ch, nxt, 4, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
        layers += [conv(nxt, 1, 4, 1, 1)]        # peta skor, tanpa sigmoid
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) pada rentang [-1, 1]."""
        return self.net(x)


def d_hinge_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Loss discriminator: dorong skor nyata > +1 dan skor palsu < -1."""
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def g_hinge_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Loss generator: naikkan skor keluaran sendiri.

    Bentuk non-saturating: gradiennya tidak menghilang meski discriminator
    sedang jauh lebih unggul.
    """
    return -d_fake.mean()
