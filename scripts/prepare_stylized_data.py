"""Generate deterministic low-poly stylized portraits for Phase 2 targets with adaptive structure."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset" / "photos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "stylized"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _list_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

def _sample_points(
    edge_map: np.ndarray,
    width: int,
    height: int,
    total_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    corners = np.array(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
        dtype=np.int32,
    )

    edge_points = np.column_stack(np.where(edge_map > 0))
    edge_points = edge_points[:, ::-1] if len(edge_points) else np.empty((0, 2), dtype=np.int32)

    usable_points = max(total_points - len(corners), 0)
    
    # FIX 1: Reduce edge dependency to 60% to prevent collapse on low-contrast faces
    edge_ratio = 0.60
    edge_count = min(int(usable_points * edge_ratio), len(edge_points))
    grid_count = max(usable_points - edge_count, 0)

    selected: list[np.ndarray] = [corners]

    if edge_count > 0:
        edge_indices = rng.choice(len(edge_points), size=edge_count, replace=False)
        selected.append(edge_points[edge_indices])

    # FIX 2: Inject a structured grid instead of pure randomness
    if grid_count > 0:
        grid_size = int(np.sqrt(grid_count))
        if grid_size > 0:
            xs = np.linspace(0, width - 1, grid_size, dtype=np.int32)
            ys = np.linspace(0, height - 1, grid_size, dtype=np.int32)
            xv, yv = np.meshgrid(xs, ys)
            grid_points = np.column_stack((xv.ravel(), yv.ravel()))
            selected.append(grid_points)
            
        # Fill any remaining point quota with random scattering
        remainder = grid_count - (grid_size**2)
        if remainder > 0:
            random_points = np.column_stack(
                [
                    rng.integers(0, width, size=remainder, endpoint=False),
                    rng.integers(0, height, size=remainder, endpoint=False),
                ]
            ).astype(np.int32)
            selected.append(random_points)

    points = np.vstack(selected)
    points = np.unique(points, axis=0)
    return points

def _triangle_mask(shape: tuple[int, int], triangle: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, triangle.astype(np.int32), 255)
    return mask

def stylize_image(
    image_bgr: np.ndarray,
    total_points: int,
    num_colors: int,
    canny_threshold1: int,
    canny_threshold2: int,
    seed: int,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    rng = np.random.default_rng(seed)

    filtered = image_bgr.copy()
    
    # FIX 4: Reduce Bilateral passes from 3 to 2 to preserve subtle facial structures
    for _ in range(2):
        filtered = cv2.bilateralFilter(filtered, 9, 75, 75)

    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    if np.count_nonzero(edges) < 500:
        edges = cv2.Canny(blurred, 30, 100)

    points = _sample_points(edges, width, height, total_points, rng)
    subdiv = cv2.Subdiv2D((0, 0, width, height))
    for x_coord, y_coord in points:
        subdiv.insert((int(x_coord), int(y_coord)))

    triangle_list = subdiv.getTriangleList()
    stylized = np.zeros_like(image_bgr)

    for triangle in triangle_list:
        pts = triangle.reshape(3, 2)
        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= width) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= height):
            continue

        polygon = np.round(pts).astype(np.int32)
        mask = _triangle_mask((height, width), polygon)
        if not np.any(mask):
            continue

        mean_bgr = cv2.mean(filtered, mask=mask)[:3]
        cv2.fillConvexPoly(stylized, polygon, tuple(int(value) for value in mean_bgr))

    pixels = stylized.reshape((-1, 3)).astype(np.float32)
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels,
        num_colors,
        None,
        criteria,
        1,
        cv2.KMEANS_PP_CENTERS,
    )
    quantized = centers[labels.flatten()].reshape(stylized.shape)
    return np.clip(quantized, 0, 255).astype(np.uint8)

def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    total_points: int,
    num_colors: int,
    canny_threshold1: int,
    canny_threshold2: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {input_dir}")

    for index, image_path in enumerate(tqdm(image_paths, desc="Preparing stylized targets")):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        stylized = stylize_image(
            image_bgr=image,
            total_points=total_points,
            num_colors=num_colors,
            canny_threshold1=canny_threshold1,
            canny_threshold2=canny_threshold2,
            seed=seed + index,
        )
        cv2.imwrite(str(output_dir / image_path.name), stylized)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic low-poly targets for Phase 2")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with source face photos")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for stylized outputs")
    parser.add_argument("--points", type=int, default=850, help="Approximate number of triangulation points")
    parser.add_argument("--colors", type=int, default=6, help="Color palette size after quantization")
    parser.add_argument("--canny-threshold1", type=int, default=80, help="Lower Canny threshold")
    parser.add_argument("--canny-threshold2", type=int, default=180, help="Upper Canny threshold")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for deterministic outputs")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        total_points=args.points,
        num_colors=args.colors,
        canny_threshold1=args.canny_threshold1,
        canny_threshold2=args.canny_threshold2,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()