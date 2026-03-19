"""Final ID-style preprocessor: Gemini detection, deterministic crop, background, and lighting."""

from __future__ import annotations

import cv2
import numpy as np
import argparse
import io
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv
from rembg import remove


def apply_lighting_adjustment(image_rgb: np.ndarray) -> np.ndarray:
    """Apply global lighting normalization to an RGB image."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    gamma = 1.45
    v_float = v.astype(np.float32) / 255.0
    v_corrected = np.power(v_float, 1.0 / gamma) * 255.0
    v = v_corrected.astype(np.uint8)

    s_float = s.astype(np.float32)
    s_corrected = np.clip(s_float * 0.85, 0, 255)
    s = s_corrected.astype(np.uint8)

    hsv_corrected = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2RGB)

def apply_skin_tone_adjustment(image_rgb: np.ndarray, strength: float = 0.75) -> np.ndarray:
    """Lighten skin regions while keeping non-skin regions mostly unchanged."""
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)

    # Broad skin range in YCrCb; soft mask avoids hard edges.
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 0)
    mask_f = (skin_mask.astype(np.float32) / 255.0) * float(np.clip(strength, 0.0, 1.0))
    mask_f = np.expand_dims(mask_f, axis=2)

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Lift luminance and reduce warm cast for a stronger "fairer" tone.
    l_lifted = np.clip(l_channel + 30.0, 0, 255)
    a_shifted = np.clip(a_channel - 3.0, 0, 255)
    b_shifted = np.clip(b_channel - 8.0, 0, 255)

    l_final = (1.0 - mask_f[:, :, 0]) * l_channel + mask_f[:, :, 0] * l_lifted
    a_final = (1.0 - mask_f[:, :, 0]) * a_channel + mask_f[:, :, 0] * a_shifted
    b_final = (1.0 - mask_f[:, :, 0]) * b_channel + mask_f[:, :, 0] * b_shifted

    lab_final = cv2.merge((l_final, a_final, b_final)).astype(np.uint8)
    return cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)

def reduce_local_skin_shadows(image_rgb: np.ndarray, strength: float = 0.50) -> np.ndarray:
    """Lift locally darker skin regions (for example around mouth/cheeks) while preserving edges."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return image_rgb

    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 0).astype(np.float32) / 255.0

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(lab)

    skin_pixels = l_channel[skin_mask > 0.15]
    if skin_pixels.size < 100:
        return image_rgb

    # Use a low-mid skin luminance anchor and only lift pixels darker than that anchor.
    anchor = np.percentile(skin_pixels, 35)
    deficit = np.clip(anchor - l_channel, 0.0, 255.0)
    deficit = cv2.GaussianBlur(deficit, (0, 0), 4.0)

    l_corrected = np.clip(l_channel + (deficit * skin_mask * (1.2 * strength)), 0.0, 255.0)

    lab_final = cv2.merge((l_corrected, a_channel, b_channel)).astype(np.uint8)
    return cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)

def process_id_photo_with_gemini(
    input_path: str,
    output_path: str,
    skin_strength: float = 0.75,
    shadow_fix_strength: float = 0.50,
) -> bool:
    load_dotenv()

    input_file = Path(input_path)
    output_file = Path(output_path)

    try:
        # Gemini face-detection portion is intentionally disabled.
        # The rest of preprocessing (background + lighting + skin tone) still runs.
        print("Gemini face detection disabled; running local preprocessing only.")

        # 1) Remove background
        print("Removing background...")
        with input_file.open("rb") as f:
            input_data = f.read()

        subject_data = remove(input_data)
        subject_image = Image.open(io.BytesIO(subject_data)).convert("RGBA")

        # 2) Apply facial/skin adjustments on subject only (before background compositing)
        subject_rgba = np.array(subject_image)
        subject_rgb = subject_rgba[:, :, :3]
        alpha = subject_rgba[:, :, 3]

        adjusted_rgb = apply_lighting_adjustment(subject_rgb)
        adjusted_rgb = reduce_local_skin_shadows(adjusted_rgb, strength=shadow_fix_strength)
        adjusted_rgb = apply_skin_tone_adjustment(adjusted_rgb, strength=skin_strength)

        adjusted_rgba = np.dstack((adjusted_rgb, alpha)).astype(np.uint8)
        adjusted_subject = Image.fromarray(adjusted_rgba, mode="RGBA")

        # 3) Composite adjusted subject onto standard blue background
        background = Image.new("RGBA", adjusted_subject.size, (105, 175, 215, 255))
        background.paste(adjusted_subject, (0, 0), adjusted_subject)

        # 4) Normalize size and save
        final_image = background.resize((256, 256), Image.Resampling.LANCZOS).convert("RGB")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_image.save(output_file)
        print(f"Success! Saved formatted image to {output_file}")
        return True
    except Exception as exc:
        print(f"Preprocessing failed: {exc}")
        return False

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess selfie into ID-style image using Gemini + rembg")
    parser.add_argument("--input", type=str, required=True, help="Path to source selfie")
    parser.add_argument("--output", type=str, required=True, help="Path to save preprocessed ID-style image")
    parser.add_argument(
        "--skin-strength",
        type=float,
        default=0.75,
        help="Skin brightening strength in [0.0, 1.0]. Try 0.80-0.90 for stronger fairness.",
    )
    parser.add_argument(
        "--shadow-fix-strength",
        type=float,
        default=0.50,
        help="Strength of local skin shadow correction in [0.0, 1.0]. Try 0.60-0.75 for cheek/mouth dark patches.",
    )
    args = parser.parse_args()

    strength = float(np.clip(args.skin_strength, 0.0, 1.0))
    shadow_strength = float(np.clip(args.shadow_fix_strength, 0.0, 1.0))
    success = process_id_photo_with_gemini(
        args.input,
        args.output,
        skin_strength=strength,
        shadow_fix_strength=shadow_strength,
    )
    if not success:
        raise SystemExit(1)

if __name__ == "__main__":
    main()