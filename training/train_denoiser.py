import os
import torch
import yaml
from torch.utils.data import DataLoader
from models import get_model
from training.datasets import MRIVolumeDataset
from training.losses import L1Loss

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for patches in loader:  # patches = (B, N,1,D,H,W)
        patches = patches.to(device)  # move to device

        noisy = patches + 0.05 * torch.randn_like(patches)
        target = patches

        noisy = noisy.view(-1,1,96,96,96)
        target = target.view(-1,1,96,96,96)

        pred = model(noisy)

        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def main(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device()

    ds = MRIVolumeDataset(
    root_dir=cfg["data_dir"],
    patch_size=tuple(cfg["patch_size"]),
    num_patches=cfg["num_patches"],
                        )

    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    model = get_model(
        cfg["model_name"],
        in_channels=1,
        out_channels=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = L1Loss()

    for epoch in range(cfg["epochs"]):
        loss = train_epoch(model, loader, loss_fn, optimizer, device)
        print(f"[Epoch {epoch+1}] Loss = {loss:.4f}")

    torch.save(model.state_dict(), "checkpoints/final_model.pth")
    print("Model saved!")

if __name__ == "__main__":
    main()


