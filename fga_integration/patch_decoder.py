from fga_integration.fga_upsampler import FGAUpsample2D

def inject_fga(vae, mode="partial", inner_dim=64):
    """mode: 'none' (baseline) | 'partial' | 'full'"""
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
        blk.upsamplers[ui] = new
        trainable += list(new.fga.parameters())
    return trainable