# DeepSketch

DeepSketch is a modular PyTorch project that trains a Pix2Pix conditional GAN to translate face photographs into pencil-style sketches.

The project is designed for local development in VS Code and GPU training on Kaggle.

## What This Project Does

- Learns paired image-to-image translation: `photo -> sketch`
- Uses the CUFS paired face photo-sketch dataset
- Trains with a U-Net generator and 70x70 PatchGAN discriminator
- Produces checkpointed models and visual sample grids during training
- Provides a separate inference script for generating sketches from new photos

## What You Get After Training

- Best model checkpoints in `checkpoints/`
- Periodic fallback checkpoints every 10 epochs
- Side-by-side visual comparisons in `samples/`:
	- input photo
	- real target sketch
	- generated sketch
- LPIPS-based model selection (best validation LPIPS is saved)

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
├── data/
│   └── dataset_loader.py        # CUFSDataset + paired Albumentations transforms
├── models/
│   ├── generator.py             # U-Net generator
│   └── discriminator.py         # PatchGAN discriminator
├── training/
│   ├── loss_functions.py        # Generator + discriminator losses
│   └── train.py                 # End-to-end training loop
├── utils/
│   ├── metrics.py               # LPIPS + SSIM helpers
│   └── image_utils.py           # denormalize, tensor_to_image, save_image_grid
├── inference/
│   └── generate_sketch.py       # Run inference from trained checkpoint
├── notebooks/
│   └── kaggle_training.ipynb    # Kaggle orchestration notebook
├── dataset/
│   ├── photos/
│   └── sketches/
├── checkpoints/                 # Auto-created during training
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

## Dataset Setup (CUFS)

Expected structure:

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

## Training

Run training:

```bash
python training/train.py --epochs 200
```

Training script behavior (`training/train.py`):

- Auto-detects device: CUDA if available, otherwise CPU
- Sets reproducible seeds (`torch`, `numpy`, `random`)
- Loads CUFS pairs from `dataset/photos` and `dataset/sketches`
- Splits dataset 90% train / 10% validation
- Uses AMP (`torch.amp.autocast`, `GradScaler`) for faster GPU training
- Optimizes discriminator first, then generator each step
- Evaluates each epoch with LPIPS on validation set
- Saves:
	- `checkpoints/best_generator.pth`
	- `checkpoints/best_discriminator.pth`
	- fallback checkpoints every 10 epochs
- Writes sample image grids to `samples/epoch_XXXX.png`

## Inference (Generate a Sketch)

Use a trained generator checkpoint to sketch a new image:

```bash
python inference/generate_sketch.py \
	--input path/to/photo.jpg \
	--output output_sketch.png \
	--checkpoint checkpoints/best_generator.pth
```

What happens:

- Input image is resized to `256x256`
- Normalized to `[-1, 1]`
- Forwarded through the generator
- Output image is converted and saved to the given path

## Kaggle Workflow

Use `notebooks/kaggle_training.ipynb` for end-to-end Kaggle execution.

The notebook performs:

1. GPU and environment check
2. Repository clone
3. Dependency installation
4. Dataset verification at `/kaggle/input/cufs-dataset-clean/`
5. Dataset symlink creation into `dataset/`
6. Training launch
7. Sample preview
8. Packaging outputs into a zip for download

## Common Commands

Train for fewer epochs (quick smoke run):

```bash
python training/train.py --epochs 1
```

Inference with default checkpoint path:

```bash
python inference/generate_sketch.py --input path/to/photo.jpg --output sketch.png
```

## Troubleshooting

- `FileNotFoundError` for dataset paths:
	- Ensure `dataset/photos` and `dataset/sketches` exist and have matching filenames.
- Poor output quality early in training:
	- This is expected; quality improves over epochs.
- No files in `checkpoints/`:
	- Verify training actually completed at least one epoch.
- Kaggle rerun conflicts on clone/symlink:
	- The notebook already removes old directories before recreating them.

## Notes

- This codebase is modular by design (not notebook-dependent training logic).
- `notebooks/kaggle_training.ipynb` orchestrates existing scripts and does not duplicate core model/training code.
