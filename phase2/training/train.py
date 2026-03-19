"""Train the Phase 2 Pix2Pix model for sketch-to-stylized-color translation."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

PHASE2_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PHASE2_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase2.data.dataset_loader import SketchToColorDataset, get_train_transforms, get_val_transforms
from phase2.models.discriminator import PatchGANDiscriminator
from phase2.models.generator import UNetGenerator
from phase2.training.loss_functions import VGGLoss, discriminator_loss, generator_loss
from phase2.utils.image_utils import tensor_to_image
from phase2.utils.metrics import PerceptualMetrics


SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 4
LR = 2e-4
BETA1 = 0.5
NUM_WORKERS = 2
PIN_MEMORY = True
DEFAULT_EPOCHS = 200

SKETCH_DIR = str(PROJECT_ROOT / "dataset" / "sketches_p1")
TARGET_DIR = str(PROJECT_ROOT / "dataset" / "stylized")
CHECKPOINT_DIR = str(PHASE2_ROOT / "checkpoints")
SAMPLE_DIR = str(PROJECT_ROOT / "samples" / "phase2")


def _seed_everything(seed: int = SEED) -> None:
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
    amp_enabled: bool,
) -> tuple[float, float]:
    generator.eval()
    lpips_scores: list[float] = []
    ssim_scores: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            sketches = batch["sketch"].to(device, non_blocking=True)
            real_targets = batch["target"].to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                fake_targets = generator(sketches)
                fake_targets = torch.clamp(fake_targets, -1.0, 1.0)
            if torch.isnan(fake_targets).any():
                print("NaN detected in generator output (validation)")
            lpips_value = metrics.compute_lpips(fake_targets, real_targets)
            ssim_value = metrics.compute_ssim(fake_targets, real_targets)
            if np.isfinite(lpips_value):
                lpips_scores.append(lpips_value)
            if np.isfinite(ssim_value):
                ssim_scores.append(ssim_value)

    generator.train()
    mean_lpips = float(np.mean(lpips_scores)) if lpips_scores else float("inf")
    mean_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0
    return mean_lpips, mean_ssim


def _save_image_grid(
    sketches: torch.Tensor,
    real_targets: torch.Tensor,
    fake_targets: torch.Tensor,
    save_path: str,
    max_images: int = 4,
) -> None:
    count = min(max_images, sketches.size(0))
    fig, axes = plt.subplots(count, 3, figsize=(12, 4 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for index in range(count):
        axes[index, 0].imshow(tensor_to_image(sketches[index]))
        axes[index, 0].set_title("Sketch")
        axes[index, 0].axis("off")

        axes[index, 1].imshow(tensor_to_image(real_targets[index]))
        axes[index, 1].set_title("Stylized Target")
        axes[index, 1].axis("off")

        axes[index, 2].imshow(tensor_to_image(fake_targets[index]))
        axes[index, 2].set_title("Generated Color")
        axes[index, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train(num_epochs: int = DEFAULT_EPOCHS, no_amp: bool = False) -> None:
    _seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not no_amp
    print(f"Using device: {device}")
    print(f"AMP enabled: {amp_enabled}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    full_dataset = SketchToColorDataset(
        sketch_dir=SKETCH_DIR,
        target_dir=TARGET_DIR,
        transform=get_train_transforms(IMAGE_SIZE),
        image_size=IMAGE_SIZE,
    )

    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    val_dataset_ref = SketchToColorDataset(
        sketch_dir=SKETCH_DIR,
        target_dir=TARGET_DIR,
        transform=get_val_transforms(IMAGE_SIZE),
        image_size=IMAGE_SIZE,
    )
    val_dataset_clean = Subset(val_dataset_ref, val_dataset.indices)
    vgg_criterion = VGGLoss(device=device)

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

    generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
    discriminator = PatchGANDiscriminator(in_channels=6).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(BETA1, 0.999))
    scaler_g = GradScaler("cuda", enabled=amp_enabled)
    scaler_d = GradScaler("cuda", enabled=amp_enabled)

    metrics = PerceptualMetrics(device=device)
    best_lpips = float("inf")

    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        valid_steps = 0
        skipped_steps = 0
        latest_sketches = None
        latest_targets = None
        latest_fake = None

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)
        for batch in progress:
            sketches = batch["sketch"].to(device, non_blocking=True)
            real_targets = batch["target"].to(device, non_blocking=True)

            opt_d.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                fake_targets = generator(sketches)
                fake_targets = torch.clamp(fake_targets, -1.0, 1.0)
                real_pred = discriminator(sketches, real_targets)
                fake_pred = discriminator(sketches, fake_targets.detach())
                d_loss, _, _ = discriminator_loss(real_pred, fake_pred)

            if torch.isnan(fake_targets).any():
                print("NaN detected in generator output")
                skipped_steps += 1
                progress.set_postfix(D="skip", G="skip")
                continue

            if not torch.isfinite(d_loss):
                skipped_steps += 1
                progress.set_postfix(D="nan", G="skip")
                continue

            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            scaler_d.step(opt_d)
            scaler_d.update()

            opt_g.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                fake_pred = discriminator(sketches, fake_targets)
                g_loss, _, _ = generator_loss(fake_pred, fake_targets, real_targets, vgg_criterion=vgg_criterion)

            if not torch.isfinite(g_loss):
                skipped_steps += 1
                progress.set_postfix(D=f"{d_loss.item():.4f}", G="nan")
                continue

            scaler_g.scale(g_loss).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            scaler_g.step(opt_g)
            scaler_g.update()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            valid_steps += 1
            latest_sketches = torch.nan_to_num(sketches.detach(), nan=0.0, posinf=1.0, neginf=-1.0).cpu()
            latest_targets = torch.nan_to_num(real_targets.detach(), nan=0.0, posinf=1.0, neginf=-1.0).cpu()
            latest_fake = torch.nan_to_num(fake_targets.detach(), nan=0.0, posinf=1.0, neginf=-1.0).cpu()
            progress.set_postfix(D=f"{d_loss.item():.4f}", G=f"{g_loss.item():.4f}")

        avg_d = epoch_d_loss / max(1, valid_steps)
        avg_g = epoch_g_loss / max(1, valid_steps)
        val_lpips, val_ssim = _validate(generator, val_loader, metrics, device, amp_enabled)
        print(
            f"  -> D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}  "
            f"val_LPIPS: {val_lpips:.4f}  val_SSIM: {val_ssim:.4f}  best_LPIPS: {best_lpips:.4f}  "
            f"valid_steps: {valid_steps}  skipped_steps: {skipped_steps}"
        )

        if val_lpips < best_lpips:
            best_lpips = val_lpips
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "best_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, "best_discriminator.pth"))
            print(f"  Saved new best model (LPIPS={val_lpips:.4f})")

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, f"generator_epoch{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{epoch}.pth"))
            print(f"  Saved fallback checkpoint at epoch {epoch}")

        if latest_sketches is not None and latest_targets is not None and latest_fake is not None:
            _save_image_grid(
                latest_sketches[:4],
                latest_targets[:4],
                latest_fake[:4],
                os.path.join(SAMPLE_DIR, f"epoch_{epoch:04d}.png"),
            )

    print("Phase 2 training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 2 sketch-to-color Pix2Pix model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (use full fp32 training).",
    )
    args = parser.parse_args()
    train(num_epochs=args.epochs, no_amp=args.no_amp)