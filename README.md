# Simple 3D Gaussian Splatting

A minimal 3DGS implementation built on [gsplat](https://github.com/nerfstudio-project/gsplat).

## Dependencies

- PyTorch
- gsplat
- torchvision
- omegaconf
- tqdm
- PyYAML

## Installation

```
conda env create -f environment.yml
conda activate gsplat
```

## Project Structure

```
3DRR_codebase/
├── config/             # Training configs (YAML)
├── core/
│   ├── data/           # Dataset loaders (Blender format)
│   ├── libs/           # Utilities (SSIM, config, etc.)
│   └── model/          # 3DGS model
├── train.py            # Training + validation + test rendering
├── eval.py             # Load checkpoint and render test set
└── outputs/            # Training outputs
```

## Data Format

Expects datasets in Blender/NeRF-synthetic JSON format with calibrated intrinsics.


## Training

```bash
python train.py -c config/cupcake.yaml
```

Outputs are saved to `outputs/<experiment>/<timestamp>/`:
- `latest.pt` — model checkpoint
- `config.yaml` — copy of training config
- `examples/` — validation render grids (`val_step*.jpg`) and augmented training samples (`train_aug.jpg`)
- `test/` — rendered test images

## Evaluation (Render Only)

Load a checkpoint and render the test set (no metrics, no GT images needed):

```bash
python eval.py -w outputs/.../latest.pt
```

Config is automatically loaded from the checkpoint directory. Rendered images are saved to `{checkpoint_dir}/test/`.


## Appearance-Branch Workflow (方案 C)

This repository now supports a NeRF-W style per-image appearance/exposure branch.

### 1) End-to-end usage

1. **Pick config** (e.g. `config/cupcake.yaml`).
2. **Train**:
   ```bash
   python train.py -c config/cupcake.yaml
   ```
3. **Find output folder** printed by training:
   `outputs/<EXP_STR>/<TIME_STR>/`
4. **Run offline rendering with checkpoint**:
   ```bash
   python eval.py -w outputs/<EXP_STR>/<TIME_STR>/latest.pt
   ```

### 2) How train/val/test behave now

- **Train loop** uses per-image `image_id` and enables the appearance branch (`canonical=False`) so exposure variations can be absorbed by `appearance_embeddings + exposure_mlp`.
- **Validation / Test rendering** force `canonical=True`, i.e. disable appearance correction and render canonical scene output.
- **Checkpoint** saves full `model.state_dict()` so both Gaussian parameters and appearance branch are persisted.

### 3) How to judge if training is healthy

Check these signals together:

1. **Loss/PSNR trend**: training logs should show decreasing loss and generally increasing PSNR.
2. **Visual spot-check**: `examples/val_step*.jpg` should become sharper and cleaner over steps.
3. **Test render sanity**: files should be continuously written under `test/` and not be blank/NaN artifacts.
4. **Densification behavior**: `num_gaussians` should increase in densification window and stabilize later.

### 4) Debug checklist (quick triage)

If results are unstable or too dark/too bright:

1. **Config toggles**
   - Set `USE_APPEARANCE: True` to enable branch.
   - Tune `LR_APPEARANCE` (start from `2.5e-3`, lower if unstable).
   - Tune `APPEARANCE_DIM` (`16/32/64`) if underfitting/overfitting.
   - `APPEARANCE_HIDDEN_DIM` controls exposure MLP capacity (default `128`; try `64` for speed, `256` for harder scenes).
   - Keep `TRAIN_TARGET_GAMMA: 1.0` when using appearance decoupling (set <1 only if you intentionally want gamma-brightened supervision).
2. **Data consistency**
   - Ensure Blender JSON intrinsics (`fl_x/fl_y/cx/cy`) match image resolution.
   - Ensure `train/val/test` image folders and `transforms_*.json` are aligned.
3. **Checkpoint compatibility**
   - If evaluating old checkpoints (without appearance branch), eval fallback still works, but appearance weights are absent by design.
   - For new checkpoints, eval infers embedding table size from checkpoint to avoid shape mismatch.
4. **Runtime environment**
   - `gsplat` needs CUDA backend for actual rasterization speed; missing CUDA can cause runtime failure in forward rendering.

### 5) Recommended minimal debug commands

```bash
# syntax check
python -m compileall train.py eval.py core/model/simple_3dgs.py core/data/blender.py

# short smoke train
python train.py -c config/cupcake_smoke.yaml

# render with trained checkpoint
python eval.py -w outputs/<EXP_STR>/<TIME_STR>/latest.pt
```
