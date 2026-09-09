import torch.nn as nn
from fga.archs.fga_arch import FGA

class FGAUpsample2D(nn.Module):
    def __init__(self, orig_upsampler, inner_dim=64):
        super().__init__()
        self.channels = orig_upsampler.channels
        self.out_channels = orig_upsampler.out_channels
        self.orig = orig_upsampler                      # jalur bawaan, dibekukan
        for p in self.orig.parameters():
            p.requires_grad_(False)

        self.fga = FGA(dim=inner_dim, back_embed_dim=self.channels,
                       out_dim=self.out_channels, upscale=2,
                       window_size=1, overlap_ratio=4)
        # Gain pada cabang residual. Float biasa, BUKAN Parameter/buffer, supaya
        # ia tidak masuk state_dict dan tidak mengubah bentuk checkpoint.
        #
        # Kenapa ini ada: modul yang terlatih ternyata mempelajari operator
        # high-pass yang MENGURANGKAN detail (korelasi -0.59 terhadap
        # highpass(baseline)). Karena delta bersifat high-pass, gain menjadi
        # kendali ketajaman saat inferensi tanpa training ulang:
        #     gain = 0   -> baseline persis
        #     gain = 1   -> perilaku terlatih (default; training selalu di sini)
        #     gain < 0   -> unsharp mask, lebih tajam dari baseline
        self.gain = 1.0
        # zero-init: di iterasi 0, output == baseline InvSR PERSIS
        nn.init.zeros_(self.fga.unembed.weight)
        nn.init.zeros_(self.fga.unembed.bias)

    def forward(self, hidden_states, output_size=None, *args, **kwargs):
        base  = self.orig(hidden_states, output_size, *args, **kwargs)
        # Jalur bawaan boleh fp16 (inferensi InvSR), tetapi FGA dilatih di float32
        # dan softmax attention + LayerNorm-nya tidak stabil di half precision.
        # Konversi di batas modul: masuk ikut dtype FGA, keluar ikut dtype base.
        fga_dtype = next(self.fga.parameters()).dtype
        if self.gain == 0.0:
            return base                                  # lewati FGA sepenuhnya
        delta = self.fga(hidden_states.to(fga_dtype))
        return base + self.gain * delta.to(base.dtype)   # residual berskala