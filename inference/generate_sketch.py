"""Generate a pencil sketch from a face photo using a trained generator."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.generator import UNetGenerator
from utils.image_utils import tensor_to_image


def load_generator(
    checkpoint_path: str,
    device: torch.device,
) -> UNetGenerator:
    """Load a trained generator from a checkpoint file."""
    generator = UNetGenerator(in_channels=3, out_channels=3)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    generator.to(device)
    generator.eval()
    return generator


def preprocess_image(
    image_path: str,
    image_size: int = 256,
) -> torch.Tensor:
    """Read an image, resize, and normalise to [-1, 1].

    Returns:
        Tensor of shape ``(1, 3, 256, 256)``.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))

    tensor = image.astype(np.float32) / 255.0
    tensor = (tensor * 2.0) - 1.0
    tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(tensor).unsqueeze(0)  # add batch dim


def generate_sketch(
    generator: UNetGenerator,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run inference and return the result as a uint8 HWC numpy image."""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = generator(input_tensor)
    return tensor_to_image(output.squeeze(0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sketch from photo")
    parser.add_argument("--input", type=str, required=True, help="Path to input photo")
    parser.add_argument("--output", type=str, default="output_sketch.png", help="Path for output sketch")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "best_generator.pth"),
        help="Path to generator checkpoint",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = load_generator(args.checkpoint, device)
    input_tensor = preprocess_image(args.input)
    sketch = generate_sketch(generator, input_tensor, device)

    # Save as BGR for OpenCV
    cv2.imwrite(args.output, cv2.cvtColor(sketch, cv2.COLOR_RGB2BGR))
    print(f"Sketch saved to {args.output}")


if __name__ == "__main__":
    main()
