import torch
from models import get_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_one(name: str, patch_size=(96, 96, 96)):
    print(f"\nTesting model: {name}")
    device = get_device()
    print(f"Using device: {device}")

    D, H, W = patch_size
    in_channels = 2
    out_channels = 1

    # Build model (only SwinUNETR needs img_size)
    if name == "swin_unetr":
        model = get_model(
            name,
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=patch_size,
        )
    else:
        model = get_model(
            name,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    model.to(device)
    model.eval()

    # Dummy input patch: (B, C, D, H, W)
    x = torch.randn(1, in_channels, D, H, W, device=device)

    with torch.no_grad():
        y = model(x)

    print(f"Input shape: {x.shape}, Output shape: {y.shape}")


if __name__ == "__main__":
    # You can comment out any model here if it misbehaves temporarily
    model_names = [
        "unet3d",
        "swin_unetr",
        "convnext3d_unet",
    ]
    for name in model_names:
        run_one(name)
