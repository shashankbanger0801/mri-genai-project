from .swin_unetr_wrapper import build_swin_unetr
from .unet3d_monai import build_unet3d
from .convnext3d_unet import build_convnext3d_unet

def get_model(name: str, **kwargs):
    name = name.lower()
    if name == "swin_unetr":
        return build_swin_unetr(**kwargs)
    elif name == "unet3d":
        return build_unet3d(**kwargs)
    elif name == "convnext3d_unet":
        return build_convnext3d_unet(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
