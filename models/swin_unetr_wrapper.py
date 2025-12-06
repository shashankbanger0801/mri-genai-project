# models/swin_unetr_wrapper.py

from monai.networks.nets import SwinUNETR


def build_swin_unetr(
    in_channels: int = 2,
    out_channels: int = 1,
    img_size = (96, 96, 96),   # must match your patch size in tests
    feature_size: int = 48,
):
    """
    Wrapper for MONAI 1.4.0 SwinUNETR.

    In MONAI 1.4.0 the signature is:

        SwinUNETR(
            img_size,
            in_channels,
            out_channels,
            feature_size=...,
            use_checkpoint=...,
            ...
        )

    We call it with positional arguments for the first three.
    """
    model = SwinUNETR(
        img_size,        # 1st positional: img_size
        in_channels,     # 2nd positional: in_channels
        out_channels,    # 3rd positional: out_channels
        feature_size=feature_size,
        use_checkpoint=False,
    )
    return model
