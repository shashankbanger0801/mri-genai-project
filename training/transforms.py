from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    RandGaussianNoised
)

def get_default_transforms():
    return Compose([
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0]),
        RandAffined(keys=["image"], prob=0.3, rotate_range=0.2),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
    ])
