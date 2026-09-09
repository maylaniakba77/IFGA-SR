from fga_integration.fga_upsampler import FGAUpsample2D

def inject_fga(vae, mode="partial", inner_dim=64, gain=1.0):
    """mode: 'none' (baseline) | 'partial' | 'full'

    gain: skala cabang residual saat inferensi. Selalu 1.0 saat training.
          0.0 = baseline persis, <0 = unsharp mask (lihat FGAUpsample2D).
    """
    if mode == "none":
        return []
    targets = [(bi, ui)
               for bi, blk in enumerate(vae.decoder.up_blocks)
               if getattr(blk, "upsamplers", None)
               for ui, _ in enumerate(blk.upsamplers)]
    if mode == "partial":
        targets = targets[-1:]          # hanya blok terakhir

    trainable = []
    for bi, ui in targets:
        blk  = vae.decoder.up_blocks[bi]
        orig = blk.upsamplers[ui]
        new  = FGAUpsample2D(orig, inner_dim).to(orig.conv.weight.device,
                                                 orig.conv.weight.dtype)
        # FGA tetap float32 apa pun dtype backbone: itulah presisi pelatihannya.
        new.fga.float()
        new.gain = float(gain)
        blk.upsamplers[ui] = new
        trainable += list(new.fga.parameters())
    return trainable