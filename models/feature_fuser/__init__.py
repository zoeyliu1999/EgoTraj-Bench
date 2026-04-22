"""
the loading function to get the fuser
1. shared: use the shared way to fuse
"""

from .shared_fuser import SharedFuser

__all__ = {
    "SharedFuser": SharedFuser,
}


def build_feature_fuser(config):
    model = __all__[config.MODEL.FUSER_NAME](config=config)
    return model
