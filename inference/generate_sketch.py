"""Generate a pencil sketch from a face photo with alignment, CLAHE, and TTA."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.generator import UNetGenerator
from utils.image_utils import tensor_to_image

def load_generator(checkpoint_path: str, device: torch.device) -> UNetGenerator:
    """Load a trained generator from a checkpoint file."""
    generator = UNetGenerator(in_channels=3, out_channels=3)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.to(device)
    generator.eval()
    return generator

def fallback_center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Fallback mechanism if no face is detected."""
    h, w, _ = image.shape
    side = min(h, w)
    cy, cx = h // 2, w // 2
    cropped = image[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]
    return cv2.resize(cropped, (size, size))

def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """Applies CLAHE to mimic flat dataset lighting without over-contrasting."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq_gray = clahe.apply(gray)
    return cv2.cvtColor(eq_gray, cv2.COLOR_GRAY2RGB)

def align_and_crop_face(image: np.ndarray, image_size: int = 256) -> np.ndarray:
    """Single-pass MediaPipe detection: rotates to level eyes and crops tightly."""
    mp_face_detection = mp.solutions.face_detection
    
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image)
        if not results.detections:
            print("Warning: No face detected by MediaPipe. Defaulting to center crop.")
            return fallback_center_crop(image, image_size)
        
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        keypoints = detection.location_data.relative_keypoints
        
        h, w, _ = image.shape
        
        right_eye = (keypoints[0].x * w, keypoints[0].y * h)
        left_eye = (keypoints[1].x * w, keypoints[1].y * h)
        # Fixed: Vector goes from left-side of screen (Right Eye) to right-side of screen (Left Eye)
        dy = left_eye[1] - right_eye[1]
        dx = left_eye[0] - right_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        box_cx = (bbox.xmin + bbox.width / 2) * w
        box_cy = (bbox.ymin + bbox.height / 2) * h
        point = np.array([[box_cx, box_cy, 1.0]])
        new_center = M.dot(point.T).T[0]
        
        cx, cy = int(new_center[0]), int(new_center[1])
        
        face_h = int(bbox.height * h)
        # Increased from 1.5 to 1.9 to zoom out and capture all hair volume
        side = int(face_h * 1.9)
        
        x1 = max(0, cx - side // 2)
        # Shifted from 0.45 to 0.50 to push the crop higher up the head
        y1 = max(0, cy - int(side * 0.50)) 

        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        
        final_side = min(x2 - x1, y2 - y1)
        cropped = rotated_image[y1:y1+final_side, x1:x1+final_side]
        
        if cropped.size == 0:
            return fallback_center_crop(rotated_image, image_size)
            
        return cv2.resize(cropped, (image_size, image_size))

def preprocess_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    """Read an image, auto-align, normalize lighting, and convert to tensor."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    aligned_face = align_and_crop_face(image, image_size)
    normalized_face = normalize_lighting(aligned_face)
    
    tensor = normalized_face.astype(np.float32) / 255.0
    tensor = (tensor * 2.0) - 1.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return torch.from_numpy(tensor).unsqueeze(0)

def generate_sketch(generator: UNetGenerator, input_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run inference with Test-Time Augmentation (TTA) and return the result."""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        pred1 = generator(input_tensor)
        
        flipped_input = torch.flip(input_tensor, dims=[3])
        pred2 = torch.flip(generator(flipped_input), dims=[3])
        
        output = (pred1 + pred2) / 2.0
        
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

    cv2.imwrite(args.output, cv2.cvtColor(sketch, cv2.COLOR_RGB2BGR))
    print(f"Sketch saved to {args.output}")

if __name__ == "__main__":
    main()