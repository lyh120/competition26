# AGENTS.md - 3DRR Codebase Agent Guidelines

This document provides guidelines for AI agents working in this repository.

## Project Overview

A minimal 3D Gaussian Splatting implementation built on [gsplat](https://github.com/nerfstudio-project/gsplat). The codebase trains 3DGS models on Blender/NeRF-synthetic format datasets and renders novel views.

## Project Structure

```
3DRR_codebase/
├── config/             # Training configs (YAML)
├── core/
│   ├── data/           # Dataset loaders (Blender format)
│   ├── libs/           # Utilities (SSIM, config, etc.)
│   └── model/          # 3DGS model (Simple3DGS)
├── train.py            # Training + validation + test rendering
├── eval.py             # Load checkpoint and render test set
├── requirements.txt    # pip dependencies
└── environment.yml     # conda environment
```

## Build/Lint/Test Commands

### Environment Setup

```bash
# Using conda
conda env create -f environment.yml
conda activate gsplat

# Using pip
pip install -r requirements.txt
```

### Running Training

```bash
python train.py -c config/cupcake.yaml
```

### Running Evaluation

```bash
python eval.py -w outputs/<experiment>/<timestamp>/latest.pt
```

### Testing

**No formal test suite exists.** To run a quick validation:
- Train for a few hundred steps and check output images in `outputs/`
- Verify rendered images appear in `examples/` and `test/` directories

### Linting/Type Checking

**No linting or type checking is configured.** If needed, install and run:
```bash
pip install ruff black mypy
ruff check .
black --check .
mypy .
```

## Code Style Guidelines

### General Conventions

- **Language**: Python 3.10+
- **Indentation**: 4 spaces (no tabs)
- **Line length**: Reasonable (under ~120 chars preferred)
- **Copyright header**: Include on new files:
  ```python
  # Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)
  ```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | CamelCase | `Simple3DGS`, `Blender`, `ConfigDict` |
| Functions/methods | snake_case | `train()`, `gamma_augment()`, `_load_data()` |
| Private attributes | Leading underscore | `_records_keys`, `_load_images` |
| Constants | UPPER_SNAKE | `NUM_INIT_POINTS` |
| Config keys | UPPER_SNAKE | `LR_MEANS`, `SH_DEGREE` |

### Imports

Organize in three sections with blank lines between:
1. Standard library (`os`, `json`, `random`, `argparse`)
2. Third-party packages (`torch`, `gsplat`, `yaml`, `omegaconf`)
3. Local modules (`from core.data import ...`)

```python
import argparse
import os
import random

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

from core.data import Blender
from core.libs import ConfigDict
from core.model import Simple3DGS
```

### Type Hints

- Use type hints for function signatures where beneficial
- Document complex parameters in docstrings using Args/Returns format:
  ```python
  def forward(self, camtoworld, img_h, img_w):
      """
      Render an image from the given camera pose.

      Args:
          camtoworld: (3, 4) camera-to-world transformation matrix.
          img_h: image height in pixels.
      Returns:
          rendered: (H, W, 3) rendered RGB image.
      """
  ```

### Error Handling

- Use assertions for internal invariants: `assert split in ["train", "val", "test"]`
- Raise descriptive exceptions: `raise FileNotFoundError(f"{path} was not found.")`
- Use `warnings.filterwarnings("ignore", ...)` for known benign warnings

### PyTorch Patterns

- Use `@torch.no_grad()` decorator for inference functions (`validate`, `evaluate`)
- Use `nn.Module` base class for models
- Use `nn.ParameterDict` for learnable parameters
- Use `.to(device)` for model/data placement
- Use `torch.no_grad()` context for evaluation

### Configuration

- Use YAML files in `config/` for hyperparameters
- Use OmegaConf/ConfigDict for config loading
- Config keys: UPPER_SNAKE case
- Save config copy to output directory during training

### Model Checkpointing

- Save to `outputs/<EXP_STR>/<TIME_STR>/latest.pt`
- Save config alongside: `outputs/.../config.yaml`
- Output structure:
  ```
  outputs/
  └── <experiment>/
      └── <timestamp>/
          ├── latest.pt
          ├── config.yaml
          ├── examples/
          │   ├── val_step*.jpg
          │   └── train_aug.jpg
          └── test/
              └── *.png
  ```

### Data Loading

- Use `torch.utils.data.Dataset` base class
- Support `train`, `val`, `test` splits
- Use `torchvision.io.read_image` for image loading
- Pre-load images with ThreadPoolExecutor for training

### Git/Version Control

- Default branch: main (check with `git branch`)
- Outputs directory is gitignored
- Do not commit: `outputs/`, `.pt` checkpoints, large data files

## Common Tasks

### Adding a New Config
Create YAML in `config/` following existing patterns:
```yaml
DATASET:
  NAME: 'MyDataset'
  DATA_PATH: '/path/to/data'
  BACKGROUND_COLOR: 255.0

MODEL:
  NAME: 'Simple3DGS'
  NUM_INIT_POINTS: 100000
  SH_DEGREE: 3
  # ... other hyperparameters
```

### Adding a New Model
1. Create `core/model/<name>.py` with `nn.Module` class
2. Register in `core/model/__init__.py`
3. Add config key handling in training script

### Debugging Training
- Check `outputs/.../examples/val_step*.jpg` for visual quality
- Monitor PSNR in training log
- Verify `num_gaussians` is increasing during densification
