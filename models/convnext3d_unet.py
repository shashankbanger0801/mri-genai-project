import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock3D(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        # depthwise conv
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # layer norm over channels
        self.norm = nn.LayerNorm(dim)
        # pointwise MLP
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        shortcut = x  # (B, C, D, H, W)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        x = self.norm(x)
        x = self.pw1(x)
        x = F.gelu(x)
        x = self.pw2(x)
        x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        return x + shortcut


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock3D(out_channels) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.down(x)
        x = self.blocks(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=2):
        super().__init__()
        # upsample from in_channels → out_channels
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        # after concat, channels = out_channels + skip_channels
        self.reduce = nn.Conv3d(
            out_channels + skip_channels, out_channels, kernel_size=1
        )
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock3D(out_channels) for _ in range(num_blocks)
        ])

    def forward(self, x, skip):
        x = self.up(x)

        # Make sure spatial sizes match (in case of odd dimensions)
        if x.shape[-3:] != skip.shape[-3:]:
            # simple center-crop or min-crop
            minD = min(x.shape[-3], skip.shape[-3])
            minH = min(x.shape[-2], skip.shape[-2])
            minW = min(x.shape[-1], skip.shape[-1])
            x = x[..., :minD, :minH, :minW]
            skip = skip[..., :minD, :minH, :minW]

        x = torch.cat([x, skip], dim=1)  # concat on channels
        x = self.reduce(x)               # project back to out_channels
        x = self.blocks(x)
        return x


class ConvNeXt3DUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_dim=32):
        super().__init__()

        # Encoder
        self.stem = nn.Conv3d(in_channels, base_dim, kernel_size=3, padding=1)

        self.enc1 = EncoderStage(base_dim, base_dim * 2)      # 32 → 64
        self.enc2 = EncoderStage(base_dim * 2, base_dim * 4)  # 64 → 128
        self.enc3 = EncoderStage(base_dim * 4, base_dim * 8)  # 128 → 256

        # Bottleneck (256 channels)
        self.bottleneck = nn.Sequential(
            ConvNeXtBlock3D(base_dim * 8),
            ConvNeXtBlock3D(base_dim * 8),
        )

        # Decoder: note skip channels
        self.dec3 = DecoderStage(
            in_channels=base_dim * 8,   # 256 from bottleneck
            skip_channels=base_dim * 4, # 128 from enc2
            out_channels=base_dim * 4,  # 128
        )
        self.dec2 = DecoderStage(
            in_channels=base_dim * 4,   # 128 from dec3
            skip_channels=base_dim * 2, # 64 from enc1
            out_channels=base_dim * 2,  # 64
        )
        self.dec1 = DecoderStage(
            in_channels=base_dim * 2,   # 64 from dec2
            skip_channels=base_dim,     # 32 from stem
            out_channels=base_dim,      # 32
        )

        # Output layer
        self.out_conv = nn.Conv3d(base_dim, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.stem(x)   # (B, 32, D, H, W)
        x1 = self.enc1(x0)  # (B, 64, D/2, H/2, W/2)
        x2 = self.enc2(x1)  # (B, 128, D/4, H/4, W/4)
        x3 = self.enc3(x2)  # (B, 256, D/8, H/8, W/8)

        # Bottleneck
        xb = self.bottleneck(x3)  # (B, 256, D/8, H/8, W/8)

        # Decoder
        d3 = self.dec3(xb, x2)  # (B, 128, D/4, H/4, W/4)
        d2 = self.dec2(d3, x1)  # (B, 64, D/2, H/2, W/2)
        d1 = self.dec1(d2, x0)  # (B, 32, D, H, W)

        out = self.out_conv(d1)  # (B, out_channels, D, H, W)
        return out


def build_convnext3d_unet(in_channels=2, out_channels=1, base_dim=32):
    return ConvNeXt3DUNet(in_channels=in_channels, out_channels=out_channels, base_dim=base_dim)
