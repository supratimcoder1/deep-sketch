"""Bare-metal inference pipeline with optional preprocessing and no TTA."""

from __future__ import annotations

import argparse
import sys
import tempfile
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
    parser.add_argument(
        "--enable-preprocess",
        action="store_true",
        help="Enable optional ID-photo preprocessing before Phase 1",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Deprecated compatibility flag; forces preprocessing off",
    )
    parser.add_argument(
        "--preprocess-output",
        type=str,
        default=None,
        help="Optional path to save the preprocessed image before Phase 1",
    )
    parser.add_argument(
        "--preprocess-skin-strength",
        type=float,
        default=0.75,
        help="Skin brightening strength passed to preprocessing in [0.0, 1.0]",
    )
    parser.add_argument(
        "--preprocess-shadow-fix-strength",
        type=float,
        default=0.50,
        help="Local skin shadow correction strength passed to preprocessing in [0.0, 1.0]",
    )
    parser.add_argument(
        "--phase1-sketch-output",
        type=str,
        default=None,
        help="Optional path to save intermediate Phase 1 sketch (restored to input dimensions)",
    )
    parser.add_argument("--phase1-checkpoint", type=str, default=str(PROJECT_ROOT / "phase1" / "checkpoints" / "best_generator.pth"))
    parser.add_argument("--phase2-checkpoint", type=str, default=str(PROJECT_ROOT / "phase2" / "checkpoints" / "best_generator.pth"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    p1_gen = load_sketch_generator(args.phase1_checkpoint, device)
    p2_gen = load_color_generator(args.phase2_checkpoint, device)

    pipeline_input = args.input
    preprocess_enabled = args.enable_preprocess and not args.skip_preprocess
    if preprocess_enabled:
        try:
            from scripts.preprocess_id_photo import process_id_photo_with_gemini
        except Exception as exc:
            raise RuntimeError(
                "Failed to import preprocessing module. Install optional deps and retry: "
                "pip install -r requirements.txt"
            ) from exc

        preprocess_output = args.preprocess_output
        if preprocess_output is None:
            tmp = Path(tempfile.gettempdir()) / "deep_sketch_preprocessed.jpg"
            preprocess_output = str(tmp)

        skin_strength = float(np.clip(args.preprocess_skin_strength, 0.0, 1.0))
        shadow_fix_strength = float(np.clip(args.preprocess_shadow_fix_strength, 0.0, 1.0))
        print("Running ID photo preprocessing...")
        ok = process_id_photo_with_gemini(
            args.input,
            preprocess_output,
            skin_strength=skin_strength,
            shadow_fix_strength=shadow_fix_strength,
        )
        if not ok:
            raise RuntimeError("ID photo preprocessing failed. Check input image and preprocess settings.")
        pipeline_input = preprocess_output
        print(f"Using preprocessed input: {pipeline_input}")
    else:
        print("Preprocessing disabled; using raw input.")

    # 1. Read raw photo
    bgr = cv2.imread(pipeline_input, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {pipeline_input}")
    
    # CUFS photos are native 200x250. Keep original dimensions for restoration.
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2. Phase 1: Photo -> Sketch
    p1_tensor = prepare_tensor(rgb)
    sketch_256 = generate_sketch(p1_gen, p1_tensor, device)
    
    # Restore to original dimensions just like we did when generating Phase 2 training data
    sketch_restored = cv2.resize(sketch_256, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
    if args.phase1_sketch_output:
        phase1_path = Path(args.phase1_sketch_output)
        phase1_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(phase1_path), cv2.cvtColor(sketch_restored, cv2.COLOR_RGB2BGR))
        print(f"Phase 1 sketch saved to {phase1_path}")

    # 3. Phase 2: Sketch -> Color
    p2_tensor = prepare_tensor(sketch_restored)
    
    # CRITICAL: TTA is explicitly forced OFF to prevent polygon ghosting
    color_256 = generate_color(p2_gen, p2_tensor, device, use_tta=False)
    
    final_output = cv2.resize(color_256, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.output, cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))
    print(f"Final output saved to {args.output}")


if __name__ == "__main__":
    main()