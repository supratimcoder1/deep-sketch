"""Pix2Pix training loop with AMP, validation, and checkpointing."""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Allow imports from repository root after phase1 layout
PHASE1_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PHASE1_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase1.data.dataset_loader import CUFSDataset, get_train_transforms, get_val_transforms
from phase1.models.generator import UNetGenerator
from phase1.models.discriminator import PatchGANDiscriminator
from phase1.training.loss_functions import generator_loss, discriminator_loss
from phase1.utils.metrics import PerceptualMetrics
from phase1.utils.image_utils import save_image_grid


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 4
LR = 2e-4
BETA1 = 0.5
NUM_WORKERS = 2
PIN_MEMORY = True
DEFAULT_EPOCHS = 200

PHOTO_DIR = str(PROJECT_ROOT / "dataset" / "photos")
SKETCH_DIR = str(PROJECT_ROOT / "dataset" / "sketches")
CHECKPOINT_DIR = str(PHASE1_ROOT / "checkpoints")
SAMPLE_DIR = str(PROJECT_ROOT / "samples")


def _seed_everything(seed: int = SEED) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate(
    generator: nn.Module,
    val_loader: DataLoader,
    metrics: PerceptualMetrics,
    device: torch.device,
) -> float:
    """Run one validation pass and return mean LPIPS."""
    generator.eval()
    lpips_scores: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            photos = batch["photo"].to(device)
            real_sketches = batch["sketch"].to(device)
            fake_sketches = generator(photos)
            score = metrics.compute_lpips(fake_sketches, real_sketches)
            lpips_scores.append(score)

    generator.train()
    return float(np.mean(lpips_scores)) if lpips_scores else float("inf")


def train(num_epochs: int = DEFAULT_EPOCHS) -> None:
    """Main training entry point."""

    _seed_everything()

    # Device -------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories --------------------------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # Dataset / DataLoaders ----------------------------------------------------
    full_dataset = CUFSDataset(
        photo_dir=PHOTO_DIR,
        sketch_dir=SKETCH_DIR,
        transform=get_train_transforms(IMAGE_SIZE),
        image_size=IMAGE_SIZE,
    )

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Override transform for validation split to avoid augmentations
    val_dataset_ref = CUFSDataset(
        photo_dir=PHOTO_DIR,
        sketch_dir=SKETCH_DIR,
        transform=get_val_transforms(IMAGE_SIZE),
        image_size=IMAGE_SIZE,
    )
    # Re-create val split with the same indices but val transforms
    val_indices = val_dataset.indices
    val_dataset_clean = torch.utils.data.Subset(val_dataset_ref, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset_clean,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"Train samples: {train_size}  |  Val samples: {val_size}")

    # Models -------------------------------------------------------------------
    generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
    discriminator = PatchGANDiscriminator(in_channels=6).to(device)

    # Optimizers ---------------------------------------------------------------
    opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

    # AMP scalers --------------------------------------------------------------
    scaler_g = GradScaler('cuda')
    scaler_d = GradScaler('cuda')

    # Metrics ------------------------------------------------------------------
    metrics = PerceptualMetrics(device=device)
    best_lpips = float("inf")

    # Training loop ------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

        for batch in pbar:
            photos = batch["photo"].to(device, non_blocking=True)
            real_sketches = batch["sketch"].to(device, non_blocking=True)

            # ----- Update Discriminator -----
            opt_d.zero_grad()
            with autocast('cuda'):
                fake_sketches = generator(photos)
                real_pred = discriminator(photos, real_sketches)
                fake_pred = discriminator(photos, fake_sketches.detach())
                d_loss, _, _ = discriminator_loss(real_pred, fake_pred)

            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ----- Update Generator -----
            opt_g.zero_grad()
            with autocast('cuda'):
                fake_pred = discriminator(photos, fake_sketches)
                g_loss, _, _ = generator_loss(fake_pred, fake_sketches, real_sketches)

            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

            pbar.set_postfix(D=f"{d_loss.item():.4f}", G=f"{g_loss.item():.4f}")

        n_batches = len(train_loader)
        avg_d = epoch_d_loss / n_batches
        avg_g = epoch_g_loss / n_batches

        # Validation -----------------------------------------------------------
        val_lpips = _validate(generator, val_loader, metrics, device)
        print(
            f"  \u2192 D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}  "
            f"val_LPIPS: {val_lpips:.4f}  best_LPIPS: {best_lpips:.4f}"
        )

        # Best checkpoint
        if val_lpips < best_lpips:
            best_lpips = val_lpips
            torch.save(
                generator.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best_generator.pth"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best_discriminator.pth"),
            )
            print(f"  \u2605 Saved new best model (LPIPS={val_lpips:.4f})")

        # Periodic fallback checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                generator.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"generator_epoch{epoch}.pth"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{epoch}.pth"),
            )
            print(f"  Saved fallback checkpoint at epoch {epoch}")

        # Save sample images each epoch
        save_image_grid(
            photos[:4].cpu(),
            real_sketches[:4].cpu(),
            fake_sketches[:4].cpu(),
            save_path=os.path.join(SAMPLE_DIR, f"epoch_{epoch:04d}.png"),
        )

    print("Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DeepSketch Pix2Pix model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    args = parser.parse_args()
    train(num_epochs=args.epochs)
