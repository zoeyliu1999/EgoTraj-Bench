from .tbd_encoder_score import ContextEncoderScore

__all__ = {
    "ContextEncoderScore": ContextEncoderScore,
}


def build_context_encoder(config, use_pre_norm):
    model = __all__[config.NAME](
        config=config, use_pre_norm=use_pre_norm
    )

    return model
