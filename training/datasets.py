import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class MRIVolumeDataset(Dataset):
    """
    Loads full 3D MRI volumes (NIfTI files) and extracts random 3D patches.
    """
    def __init__(self, root_dir, patch_size=(96, 96, 96), num_patches=4):
        super().__init__()
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Recursively walk the directory tree and collect all .nii / .nii.gz files
        self.files = []
        for r, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".nii") or f.endswith(".nii.gz"):
                    self.files.append(os.path.join(r, f))

        if len(self.files) == 0:
            print(f"[MRIVolumeDataset] WARNING: No NIfTI files found in {root_dir}")

    def load_nifti(self, path):
        img = nib.load(path)
        vol = img.get_fdata().astype(np.float32)
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)  # normalize 0–1
        return vol

    def extract_patch(self, vol):
        D, H, W = vol.shape
        pd, ph, pw = self.patch_size

        # random indices
        d = np.random.randint(0, D - pd)
        h = np.random.randint(0, H - ph)
        w = np.random.randint(0, W - pw)

        return vol[d:d+pd, h:h+ph, w:w+pw]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vol = self.load_nifti(self.files[idx])

        patches = []
        for _ in range(self.num_patches):
            patch = self.extract_patch(vol)
            patches.append(patch)

        patches = np.stack(patches, axis=0)  # (num_patches, D,H,W)
        patches = torch.from_numpy(patches).unsqueeze(1)  # → (N,1,D,H,W)
        return patches
