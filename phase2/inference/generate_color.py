"""Generate a stylized color portrait from a sketch using the Phase 2 generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PHASE2_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PHASE2_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase2.models.generator import UNetGenerator
from phase2.utils.image_utils import tensor_to_image


def load_generator(checkpoint_path: str, device: torch.device) -> UNetGenerator:
    generator = UNetGenerator(in_channels=3, out_channels=3)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.to(device)
    generator.eval()
    return generator


def preprocess_sketch(image_path: str, image_size: int = 256) -> torch.Tensor:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = image.astype(np.float32) / 255.0
    tensor = (tensor * 2.0) - 1.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return torch.from_numpy(tensor).unsqueeze(0)


def generate_color(
    generator: UNetGenerator,
    input_tensor: torch.Tensor,
    device: torch.device,
    use_tta: bool = True,
) -> np.ndarray:
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        prediction = generator(input_tensor)
        if use_tta:
            flipped_input = torch.flip(input_tensor, dims=[3])
            flipped_prediction = torch.flip(generator(flipped_input), dims=[3])
            prediction = (prediction + flipped_prediction) / 2.0
    return tensor_to_image(prediction.squeeze(0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stylized color portrait from sketch")
    parser.add_argument("--input", type=str, required=True, help="Path to input sketch")
    parser.add_argument("--output", type=str, default="output_color.png", help="Path for output portrait")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PHASE2_ROOT / "checkpoints" / "best_generator.pth"),
        help="Path to generator checkpoint",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Inference resolution")
    parser.add_argument("--no-tta", action="store_true", help="Disable horizontal flip test-time augmentation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = load_generator(args.checkpoint, device)
    input_tensor = preprocess_sketch(args.input, image_size=args.image_size)
    output_image = generate_color(generator, input_tensor, device, use_tta=not args.no_tta)

    cv2.imwrite(args.output, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Stylized portrait saved to {args.output}")


if __name__ == "__main__":
    main()