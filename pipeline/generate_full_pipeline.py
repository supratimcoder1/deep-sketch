"""Run the full photo -> sketch -> stylized color pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase1.inference.generate_sketch import generate_sketch, load_generator as load_sketch_generator, preprocess_image
from phase2.inference.generate_color import generate_color, load_generator as load_color_generator


def _sketch_array_to_tensor(sketch_rgb: np.ndarray) -> torch.Tensor:
    tensor = sketch_rgb.astype(np.float32) / 255.0
    tensor = (tensor * 2.0) - 1.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return torch.from_numpy(tensor).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full DeepSketch two-stage pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input face photo")
    parser.add_argument("--output", type=str, default="full_pipeline_color.png", help="Path to final stylized output")
    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "phase1" / "checkpoints" / "best_generator.pth"),
        help="Phase 1 generator checkpoint",
    )
    parser.add_argument(
        "--phase2-checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "phase2" / "checkpoints" / "best_generator.pth"),
        help="Phase 2 generator checkpoint",
    )
    parser.add_argument(
        "--save-intermediate-sketch",
        type=str,
        default="",
        help="Optional path to save the intermediate generated sketch",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    phase1_generator = load_sketch_generator(args.phase1_checkpoint, device)
    phase2_generator = load_color_generator(args.phase2_checkpoint, device)

    photo_tensor = preprocess_image(args.input)
    sketch_rgb = generate_sketch(phase1_generator, photo_tensor, device)
    if args.save_intermediate_sketch:
        cv2.imwrite(args.save_intermediate_sketch, cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2BGR))

    sketch_tensor = _sketch_array_to_tensor(sketch_rgb)
    color_rgb = generate_color(phase2_generator, sketch_tensor, device, use_tta=True)
    cv2.imwrite(args.output, cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
    print(f"Final stylized portrait saved to {args.output}")


if __name__ == "__main__":
    main()