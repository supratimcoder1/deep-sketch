"""Run the full photo -> sketch -> stylized color pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase1.inference.generate_sketch import generate_sketch, load_generator as load_sketch_generator
from phase2.inference.generate_color import generate_color, load_generator as load_color_generator


def preprocess_to_cufs_style(image: np.ndarray) -> np.ndarray:
    """Apply CUFS-style preprocessing: Crop, Normalize Lighting, Remove BG."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    image = image.astype(np.uint8, copy=False)
    height, width = image.shape[:2]

    # 1) Face detection + padded crop (Do this first to isolate the face)
    mp_face = mp.solutions.face_detection
    x1, y1, x2, y2 = 0, 0, width, height
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        detect_result = detector.process(image)

    if detect_result.detections:
        bbox = detect_result.detections[0].location_data.relative_bounding_box
        bx, by = bbox.xmin * width, bbox.ymin * height
        bw, bh = bbox.width * width, bbox.height * height

        pad_x, pad_top, pad_bottom = 0.35 * bw, 0.50 * bh, 0.25 * bh
        x1 = int(max(0, np.floor(bx - pad_x)))
        y1 = int(max(0, np.floor(by - pad_top)))
        x2 = int(min(width, np.ceil(bx + bw + pad_x)))
        y2 = int(min(height, np.ceil(by + bh + pad_bottom)))

        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, width, height

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = image.copy()

    # 2) CLAHE lighting normalization (Apply ONLY to natural pixels, before masking)
    lab = cv2.cvtColor(cropped, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    normalized = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # 3) Foreground segmentation + CUFS-accurate pale blue background
    mp_selfie = mp.solutions.selfie_segmentation
    # Sampled directly from the AR/CUFS dataset: a muted, pale cyan-blue
    cufs_bg_color = np.full_like(normalized, (205, 220, 225), dtype=np.uint8)
    
    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
        seg_result = segmenter.process(normalized)
    
    if seg_result.segmentation_mask is None:
        composited = normalized
    else:
        # Threshold and add a slight blur to the mask to soften harsh cutout edges
        fg_mask = (seg_result.segmentation_mask > 0.5).astype(np.float32)
        fg_mask_blurred = cv2.GaussianBlur(fg_mask, (3, 3), 0)[..., None]
        composited = (normalized.astype(np.float32) * fg_mask_blurred + 
                      cufs_bg_color.astype(np.float32) * (1.0 - fg_mask_blurred)).astype(np.uint8)
    return composited


def prepare_tensor_for_model(image: np.ndarray) -> torch.Tensor:
    """Takes any array, squashes it to 256x256 for the GAN, and converts to tensor."""
    resized = cv2.resize(image, (256, 256))
    tensor = resized.astype(np.float32) / 255.0
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

    # 1. Read and isolate the human in 200x250 space
    input_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if input_bgr is None:
        raise FileNotFoundError(f"Cannot read input image: {args.input}")
    input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    cufs_200x250 = preprocess_to_cufs_style(input_rgb)

    # 2. Squash to 256x256 and run Phase 1
    p1_tensor = prepare_tensor_for_model(cufs_200x250)
    sketch_256 = generate_sketch(phase1_generator, p1_tensor, device)

    # 3. Emulate saving to disk by shrinking back to 200x250
    sketch_200x250 = cv2.resize(sketch_256, (200, 250), interpolation=cv2.INTER_AREA)
    if args.save_intermediate_sketch:
        cv2.imwrite(args.save_intermediate_sketch, cv2.cvtColor(sketch_200x250, cv2.COLOR_RGB2BGR))

    # 4. Emulate Phase 2 dataloader by squashing to 256x256 and run Phase 2
    p2_tensor = prepare_tensor_for_model(sketch_200x250)
    color_256 = generate_color(phase2_generator, p2_tensor, device, use_tta=False)

    # 5. Restore physical aspect ratio to 200x250 and save
    final_color_200x250 = cv2.resize(color_256, (200, 250), interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.output, cv2.cvtColor(final_color_200x250, cv2.COLOR_RGB2BGR))
    
    print(f"Final stylized portrait saved to {args.output}")


if __name__ == "__main__":
    main()