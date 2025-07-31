from contextlib import contextmanager
from HYPIR.utils.tiled_vae.vaehook import VAEHook


@contextmanager
def enable_tiled_vae(
    vae,
    is_decoder,
    tile_size=256,
    dtype=None,
):
    if not is_decoder:
        original_forward = vae.encoder.forward
        model = vae.encoder
    else:
        original_forward = vae.decoder.forward
        model = vae.decoder
    model.original_forward = original_forward

    model.forward = VAEHook(
        model,
        tile_size,
        is_decoder=is_decoder,
        fast_decoder=False,
        fast_encoder=False,
        color_fix=False,
        to_gpu=False,
        dtype=dtype,
    )

    try:
        yield
    finally:
        del model.original_forward
        model.forward = original_forward
