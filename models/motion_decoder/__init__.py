from .mtr_decoder_score import MotionDecoderScore

__all__ = {
    "MotionDecoderScore": MotionDecoderScore,
}


def build_decoder(config, use_pre_norm, **kwargs):
    model = __all__[config.NAME](config=config, use_pre_norm=use_pre_norm, **kwargs)

    return model
