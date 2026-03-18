"""Generate Phase 1 sketches for Phase 2 training dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

# --- Project root setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Import Phase 1 inference functions ---
from phase1.inference.generate_sketch import (
    load_generator,
    preprocess_image,
    generate_sketch,
)

# --- Paths ---
INPUT_DIR = PROJECT_ROOT / "dataset" / "photos"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "sketches_p1"
CHECKPOINT_PATH = PROJECT_ROOT / "phase1" / "checkpoints" / "best_generator.pth"

# --- Supported formats ---
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(INPUT_DIR)
    if not image_paths:
        raise RuntimeError("No images found in dataset/photos")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Phase 1 generator...")
    generator = load_generator(str(CHECKPOINT_PATH), device)

    print("Loading Phase 1 generator...")
    generator = load_generator(str(CHECKPOINT_PATH), device)
    generator.eval()  # CRITICAL: Set to evaluation mode

    print(f"Generating sketches for {len(image_paths)} images...")

    with torch.no_grad():  # CRITICAL: Disable gradient calculation to prevent VRAM leaks
        for img_path in tqdm(image_paths, desc="Generating Phase1 sketches"):
            # --- Preprocess photo ---
            input_tensor = preprocess_image(str(img_path))

            # --- Generate sketch ---
            sketch_rgb = generate_sketch(generator, input_tensor, device)

            # --- Save ---
            save_path = OUTPUT_DIR / img_path.name
            cv2.imwrite(str(save_path), cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()