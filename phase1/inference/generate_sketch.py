"""Generate a pencil sketch from a photo with minimal preprocessing."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PHASE1_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PHASE1_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase1.models.generator import UNetGenerator
from phase1.utils.image_utils import tensor_to_image

def load_generator(checkpoint_path: str, device: torch.device) -> UNetGenerator:
    """Load a trained generator from a checkpoint file."""
    generator = UNetGenerator(in_channels=3, out_channels=3)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.to(device)
    generator.eval()
    return generator

def preprocess_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    """Read an image and convert to model tensor using only required transforms."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

    tensor = image.astype(np.float32) / 255.0
    tensor = (tensor * 2.0) - 1.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return torch.from_numpy(tensor).unsqueeze(0)

def generate_sketch(generator: UNetGenerator, input_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run a single forward pass and return the sketch."""
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
        default=str(PHASE1_ROOT / "checkpoints" / "best_generator.pth"),
        help="Path to generator checkpoint",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = load_generator(args.checkpoint, device)
    input_tensor = preprocess_image(args.input)
    sketch = generate_sketch(generator, input_tensor, device)

    cv2.imwrite(args.output, cv2.cvtColor(sketch, cv2.COLOR_RGB2BGR))
    print(f"Sketch saved to {args.output}")

if __name__ == "__main__":
    main()