# DeepSketch

DeepSketch is a modular two-stage PyTorch project built around Pix2Pix conditional GANs for portrait stylization.

The project is designed for local development in VS Code and GPU training on Kaggle.

## What This Project Does

- Phase 1 learns paired image-to-image translation: `photo -> sketch`
- Phase 2 learns paired image-to-image translation: `sketch -> stylized color portrait`
- Uses the CUFS paired face photo-sketch dataset plus deterministic stylized targets generated from the photo set
- Trains each stage with a U-Net generator and 70x70 PatchGAN discriminator
- Produces checkpointed models and visual sample grids during training
- Provides separate inference scripts for Phase 1, Phase 2, and the full chained pipeline

## What You Get After Training

- Best model checkpoints in `phase1/checkpoints/` and `phase2/checkpoints/`
- Periodic fallback checkpoints every 10 epochs for both stages
- Side-by-side visual comparisons in `samples/`:
	- Phase 1 grids in `samples/epoch_XXXX.png`
	- Phase 2 grids in `samples/phase2/epoch_XXXX.png`
- Deterministic low-poly targets in `dataset/stylized/`
- LPIPS-based model selection with SSIM reporting during validation
- An optional end-to-end script that chains both generators: `photo -> sketch -> stylized color`

## Core Architecture

| Component | Details |
|---|---|
| Generator | Pix2Pix U-Net (encoder-decoder with skip connections) |
| Discriminator | 70x70 PatchGAN |
| Input/Output Resolution | `256 x 256` |
| Input Channels | 3 (RGB photo) |
| Output Channels | 3 (sketch in RGB-compatible format) |
| Normalization | `[-1, 1]` |
| Optimizer | Adam (`lr=2e-4`, `beta1=0.5`) |
| Batch Size | 4 |

## Repository Layout

```text
deep-sketch/
├── dataset/
│   ├── photos/
│   ├── sketches/
│   └── stylized/               # Auto-generated deterministic Phase 2 targets
├── pipeline/
│   └── generate_full_pipeline.py
├── phase1/
│   ├── checkpoints/             # Auto-created during training
│   ├── data/
│   │   └── dataset_loader.py    # CUFSDataset + paired Albumentations transforms
│   ├── inference/
│   │   └── generate_sketch.py   # Run inference from trained checkpoint
│   ├── models/
│   │   ├── generator.py         # U-Net generator
│   │   └── discriminator.py     # PatchGAN discriminator
│   ├── notebooks/
│   │   └── kaggle_training.ipynb
│   ├── training/
│   │   ├── loss_functions.py    # Generator + discriminator losses
│   │   └── train.py             # End-to-end training loop
│   └── utils/
│       ├── metrics.py           # LPIPS + SSIM helpers
│       └── image_utils.py       # denormalize, tensor_to_image, save_image_grid
├── phase2/
│   ├── checkpoints/             # Auto-created during training
│   ├── data/
│   │   └── dataset_loader.py    # SketchToColorDataset + paired transforms
│   ├── inference/
│   │   └── generate_color.py    # Sketch -> stylized portrait inference
│   ├── models/
│   │   ├── generator.py         # Wrapper around Phase 1 U-Net generator
│   │   └── discriminator.py     # Wrapper around Phase 1 PatchGAN discriminator
│   ├── notebooks/
│   │   └── kaggle_phase2.ipynb
│   ├── training/
│   │   ├── loss_functions.py    # Wrapper around Phase 1 loss functions
│   │   └── train.py             # Phase 2 training loop
│   └── utils/
│       ├── metrics.py           # Wrapper around Phase 1 metrics
│       └── image_utils.py       # Wrapper around Phase 1 image utilities
├── scripts/
│   └── prepare_stylized_data.py
├── samples/                     # Auto-created during training
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10
- CUDA GPU strongly recommended for training

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies in `requirements.txt`:

- `torch`
- `torchvision`
- `albumentations`
- `opencv-python`
- `lpips`
- `torchmetrics`
- `numpy`
- `matplotlib`
- `tqdm`
- `mediapipe`

## Dataset Setup (CUFS)

Expected base structure:

```text
dataset/
├── photos/
│   ├── f-005-01.jpg
│   ├── m-008-01.jpg
│   └── ...
└── sketches/
		├── f-005-01.jpg
		├── m-008-01.jpg
		└── ...
```

Important rules:

- Filenames must match exactly between `photos/` and `sketches/`
- The dataset loader sorts filenames for deterministic pairing
- Spatial augmentations are synchronized on both images using Albumentations `additional_targets`
- Photo-only augmentations are applied only to photos

Generate Phase 2 targets from the photo directory:

```bash
python scripts/prepare_stylized_data.py --input-dir dataset/photos --output-dir dataset/stylized
```

The stylization script applies:

- bilateral filtering twice for edge-preserving smoothing
- grayscale + Gaussian blur + Canny edge detection
- deterministic point sampling with 85% edge points and 15% random points plus image corners
- Delaunay triangulation via `cv2.Subdiv2D`
- per-triangle mean-color fills
- deterministic K-means color quantization

## Training

Run training:

```bash
python phase1/training/train.py --epochs 200
```

Run Phase 2 training after generating `dataset/stylized/`:

```bash
python phase2/training/train.py --epochs 200
```

Training script behavior (`phase1/training/train.py`):

- Auto-detects device: CUDA if available, otherwise CPU
- Sets reproducible seeds (`torch`, `numpy`, `random`)
- Loads CUFS pairs from `dataset/photos` and `dataset/sketches`
- Splits dataset 90% train / 10% validation
- Uses AMP (`torch.amp.autocast`, `GradScaler`) for faster GPU training
- Optimizes discriminator first, then generator each step
- Evaluates each epoch with LPIPS on validation set
- Saves:
	- `phase1/checkpoints/best_generator.pth`
- Writes sample image grids to `samples/epoch_XXXX.png`

Phase 2 training script behavior (`phase2/training/train.py`):

- Loads paired data from `dataset/sketches` and `dataset/stylized`
- Reuses the same generator, discriminator, losses, metrics, and image utilities from Phase 1 through thin wrappers
- Tracks both LPIPS and SSIM on the validation split
- Saves:
	- `phase2/checkpoints/best_generator.pth`
- Writes sample grids to `samples/phase2/epoch_XXXX.png`

## Inference (Generate a Sketch)

Use a trained generator checkpoint to sketch a new image:

```bash
python phase1/inference/generate_sketch.py \
	--input path/to/photo.jpg \
	--output output_sketch.png \
	--checkpoint phase1/checkpoints/best_generator.pth
```

What happens:

- Input image is resized to `256x256`
- Normalized to `[-1, 1]`
- Forwarded through the generator
- Output image is converted and saved to the given path

## Inference (Generate a Stylized Portrait)

```bash
python phase2/inference/generate_color.py \
	--input path/to/sketch.png \
	--output output_color.png \
	--checkpoint phase2/checkpoints/best_generator.pth
```

This script resizes the sketch to `256x256`, normalizes it to `[-1, 1]`, runs the Phase 2 generator, and writes a stylized color portrait. Horizontal-flip TTA is enabled by default.

## Full Two-Stage Pipeline

```bash
python pipeline/generate_full_pipeline.py \
	--input path/to/photo.jpg \
	--output final_portrait.png \
	--save-intermediate-sketch intermediate_sketch.png
```

This runs Phase 1 and Phase 2 back to back in one command.

## Kaggle Workflow

Use `phase1/notebooks/kaggle_training.ipynb` for end-to-end Kaggle execution.

Use `phase2/notebooks/kaggle_phase2.ipynb` for the Phase 2 Kaggle workflow.

The notebook performs:

1. GPU and environment check
2. Repository clone
3. Dependency installation
4. Dataset verification at `/kaggle/input/cufs-dataset-clean/`
5. Dataset symlink creation into `dataset/`
6. Training launch
7. Sample preview
8. Packaging the best model into a zip for download

The Phase 2 notebook performs:

1. GPU and environment check
2. Repository clone
3. Dependency installation
4. Dataset symlink creation for `photos/` and `sketches/`
5. Stylized target generation into `dataset/stylized/`
6. Phase 2 training launch
7. Inference preview from a sample sketch
8. Packaging `phase2/checkpoints` and `samples/phase2`

## Common Commands

Train for fewer epochs (quick smoke run):

```bash
python phase1/training/train.py --epochs 1
```

Prepare stylized targets only:

```bash
python scripts/prepare_stylized_data.py --points 1200 --colors 12 --seed 42
```

Quick Phase 2 smoke run:

```bash
python phase2/training/train.py --epochs 1
```

Inference with default checkpoint path:

```bash
python phase1/inference/generate_sketch.py --input path/to/photo.jpg --output sketch.png
```

Run the full pipeline with default checkpoints:

```bash
python pipeline/generate_full_pipeline.py --input path/to/photo.jpg --output portrait.png
```

## Troubleshooting

- `FileNotFoundError` for dataset paths:
	- Ensure `dataset/photos` and `dataset/sketches` exist and have matching filenames.
- `FileNotFoundError` for Phase 2 training:
	- Ensure `dataset/stylized` has been generated and contains filenames matching `dataset/sketches`.
- Poor output quality early in training:
	- This is expected; quality improves over epochs.
- No files in checkpoint folders:
	- Verify training actually completed at least one epoch and check `phase1/checkpoints/` or `phase2/checkpoints/`.
- Kaggle rerun conflicts on clone/symlink:
	- The notebook already removes old directories before recreating them.

## Notes

- This codebase is modular by design (not notebook-dependent training logic).
- `phase1/notebooks/kaggle_training.ipynb` orchestrates existing scripts and does not duplicate core model/training code.
- `phase2/notebooks/kaggle_phase2.ipynb` follows the same pattern for Phase 2.
- Phase 2 intentionally reuses Phase 1 model and utility implementations instead of duplicating them.
