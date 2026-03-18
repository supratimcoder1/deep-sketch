"""Bare-metal inference pipeline. No preprocessing, no TTA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase1.inference.generate_sketch import generate_sketch, load_generator as load_sketch_generator
from phase2.inference.generate_color import generate_color, load_generator as load_color_generator


def prepare_tensor(image: np.ndarray) -> torch.Tensor:
    """Strictly mimics the training dataloader resizing and normalization."""
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor * 2.0) - 1.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return torch.from_numpy(tensor).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bare-metal DeepSketch pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input CUFS photo")
    parser.add_argument("--output", type=str, default="pure_pipeline_color.png", help="Path to final output")
    parser.add_argument("--phase1-checkpoint", type=str, default=str(PROJECT_ROOT / "phase1" / "checkpoints" / "best_generator.pth"))
    parser.add_argument("--phase2-checkpoint", type=str, default=str(PROJECT_ROOT / "phase2" / "checkpoints" / "best_generator.pth"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    p1_gen = load_sketch_generator(args.phase1_checkpoint, device)
    p2_gen = load_color_generator(args.phase2_checkpoint, device)

    # 1. Read raw photo
    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {args.input}")
    
    # CUFS photos are native 200x250. Keep original dimensions for restoration.
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2. Phase 1: Photo -> Sketch
    p1_tensor = prepare_tensor(rgb)
    sketch_256 = generate_sketch(p1_gen, p1_tensor, device)
    
    # Restore to original dimensions just like we did when generating Phase 2 training data
    sketch_restored = cv2.resize(sketch_256, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

    # 3. Phase 2: Sketch -> Color
    p2_tensor = prepare_tensor(sketch_restored)
    
    # CRITICAL: TTA is explicitly forced OFF to prevent polygon ghosting
    color_256 = generate_color(p2_gen, p2_tensor, device, use_tta=False)
    
    final_output = cv2.resize(color_256, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.output, cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))
    print(f"Final output saved to {args.output}")


if __name__ == "__main__":
    main()