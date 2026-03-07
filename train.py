#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import argparse
import math
import os
import random
import warnings

import gsplat
import torch
import yaml
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from core.data import Blender
from core.libs import ConfigDict, lowlight_enhance, ssim
from core.model import Simple3DGS


def prepare_train_target(image, model_cfg):
    """Prepare supervision target image for training.

    By default we supervise with raw low-light observations (clamped to [0, 1]).
    Optional gamma correction can be enabled via TRAIN_TARGET_GAMMA (>0).
    """
    gamma = float(getattr(model_cfg, "TRAIN_TARGET_GAMMA", 1.0))
    image = torch.clamp(image, 0, 1)
    if abs(gamma - 1.0) < 1e-8:
        return image
    return image.pow(gamma)


def image_gradient_l1(pred_hwc, gt_hwc):
    """Edge-aware L1 loss on horizontal/vertical image gradients."""
    pred_dx = pred_hwc[:, 1:, :] - pred_hwc[:, :-1, :]
    gt_dx = gt_hwc[:, 1:, :] - gt_hwc[:, :-1, :]
    pred_dy = pred_hwc[1:, :, :] - pred_hwc[:-1, :, :]
    gt_dy = gt_hwc[1:, :, :] - gt_hwc[:-1, :, :]
    return torch.abs(pred_dx - gt_dx).mean() + torch.abs(pred_dy - gt_dy).mean()


def validate_model_cfg(model_cfg):
    """Run lightweight config validation to fail fast on invalid settings."""
    train_gamma = float(getattr(model_cfg, "TRAIN_TARGET_GAMMA", 1.0))
    if train_gamma <= 0:
        raise ValueError(f"TRAIN_TARGET_GAMMA must be > 0, got {train_gamma}.")

    lambda_edge = float(getattr(model_cfg, "LAMBDA_EDGE", 0.0))
    if lambda_edge < 0:
        raise ValueError(f"LAMBDA_EDGE must be >= 0, got {lambda_edge}.")

    if bool(getattr(model_cfg, "USE_ADAPTIVE_RENDER_ENHANCE", True)):
        render_eps = float(getattr(model_cfg, "RENDER_EPS", 1e-6))
        target_mean = float(getattr(model_cfg, "RENDER_TARGET_MEAN", 0.5))
        max_scale = float(getattr(model_cfg, "RENDER_MAX_SCALE", 4.0))
        gamma_min = float(getattr(model_cfg, "RENDER_GAMMA_MIN", 0.4))
        gamma_max = float(getattr(model_cfg, "RENDER_GAMMA_MAX", 1.2))
        if render_eps <= 0:
            raise ValueError(f"RENDER_EPS must be > 0, got {render_eps}.")
        if target_mean <= 0 or target_mean > 1:
            raise ValueError(f"RENDER_TARGET_MEAN must be in (0, 1], got {target_mean}.")
        if max_scale <= 0:
            raise ValueError(f"RENDER_MAX_SCALE must be > 0, got {max_scale}.")
        if gamma_min <= 0 or gamma_max <= 0:
            raise ValueError("RENDER_GAMMA_MIN and RENDER_GAMMA_MAX must be > 0.")
        if gamma_min > gamma_max:
            raise ValueError("RENDER_GAMMA_MIN must be <= RENDER_GAMMA_MAX.")
    else:
        render_gamma = float(getattr(model_cfg, "RENDER_GAMMA", 1.0))
        if render_gamma <= 0:
            raise ValueError(f"RENDER_GAMMA must be > 0, got {render_gamma}.")

    if bool(getattr(model_cfg, "USE_APPEARANCE", True)):
        appearance_dim = int(getattr(model_cfg, "APPEARANCE_DIM", 32))
        hidden_dim = int(getattr(model_cfg, "APPEARANCE_HIDDEN_DIM", 128))
        appearance_lr = float(getattr(model_cfg, "LR_APPEARANCE", model_cfg.LR_SH0))
        if appearance_dim <= 0:
            raise ValueError(f"APPEARANCE_DIM must be > 0, got {appearance_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"APPEARANCE_HIDDEN_DIM must be > 0, got {hidden_dim}.")
        if appearance_lr <= 0:
            raise ValueError(f"LR_APPEARANCE must be > 0, got {appearance_lr}.")

    total_steps = int(model_cfg.TRAIN_TOTAL_STEP)
    densify_start = int(model_cfg.DENSIFY_START_STEP)
    densify_stop = int(model_cfg.DENSIFY_STOP_STEP)
    sh_upgrade_interval = int(model_cfg.SH_UPGRADE_INTERVAL)
    sh_degree = int(model_cfg.SH_DEGREE)

    if total_steps <= 0:
        raise ValueError(f"TRAIN_TOTAL_STEP must be > 0, got {total_steps}.")
    if densify_start < 0:
        raise ValueError(f"DENSIFY_START_STEP must be >= 0, got {densify_start}.")
    if densify_stop <= densify_start:
        raise ValueError(
            f"DENSIFY_STOP_STEP must be > DENSIFY_START_STEP, got {densify_stop} <= {densify_start}."
        )
    if densify_stop > total_steps:
        raise ValueError(f"DENSIFY_STOP_STEP must be <= TRAIN_TOTAL_STEP, got {densify_stop} > {total_steps}.")

    densify_ratio = densify_stop / float(total_steps)
    if densify_ratio < 0.2:
        warnings.warn(
            "DENSIFY_STOP_STEP is less than 20% of TRAIN_TOTAL_STEP. "
            "For long training, consider extending densification (e.g. 20%-35% of total steps).",
            stacklevel=2,
        )

    if sh_upgrade_interval <= 0:
        raise ValueError(f"SH_UPGRADE_INTERVAL must be > 0, got {sh_upgrade_interval}.")
    if sh_degree > 0 and sh_upgrade_interval * sh_degree > total_steps:
        warnings.warn(
            "SH_UPGRADE_INTERVAL is large relative to TRAIN_TOTAL_STEP; SH may not reach the configured maximum degree.",
            stacklevel=2,
        )



def train(config_path, device="cuda"):
    # build config
    meta_cfg = ConfigDict(config_path=config_path)
    print(meta_cfg)
    cfg = meta_cfg.MODEL
    validate_model_cfg(cfg)

    # build output directory
    output_dir = os.path.join("outputs", meta_cfg.EXP_STR, meta_cfg.TIME_STR)
    os.makedirs(os.path.join(output_dir, "examples"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(dict(meta_cfg), f, default_flow_style=False)

    # load dataset
    train_dataset = Blender(meta_cfg.DATASET, split="train")
    val_dataset = Blender(meta_cfg.DATASET, split="val", load_images=False)
    num_train = len(train_dataset._records_keys)

    # build model
    data_info = dict(train_dataset._data_info)
    data_info["num_train_images"] = train_dataset._data_info["num_images"]
    model = Simple3DGS(cfg, data_info).to(device)
    print(f"Initialized {model.num_gaussians} Gaussians")

    # per-parameter optimizers (required by gsplat DefaultStrategy)
    lr_map = {
        "means": cfg.LR_MEANS,
        "quats": cfg.LR_QUATS,
        "scales": cfg.LR_SCALES,
        "opacities": cfg.LR_OPACITIES,
        "sh0": cfg.LR_SH0,
        "shN": cfg.LR_SHN,
    }
    optimizers = {}
    for name, param in model.splats.items():
        optimizers[name] = torch.optim.Adam([param], lr=lr_map[name], eps=1e-15)
    if model.use_appearance:
        appearance_lr = getattr(cfg, "LR_APPEARANCE", cfg.LR_SH0)
        optimizers["appearance_embeddings"] = torch.optim.Adam(
            model.appearance_embeddings.parameters(), lr=appearance_lr, eps=1e-15
        )
        optimizers["exposure_mlp"] = torch.optim.Adam(
            model.exposure_mlp.parameters(), lr=appearance_lr, eps=1e-15
        )

    # exponential LR decay for means
    total_steps = cfg.TRAIN_TOTAL_STEP
    lr_final_factor = cfg.LR_MEANS_FINAL / cfg.LR_MEANS
    schedulers = {
        "means": torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=lr_final_factor ** (1.0 / total_steps)
        )
    }

    # densification strategy
    strategy = gsplat.DefaultStrategy(
        verbose=True,
        refine_start_iter=cfg.DENSIFY_START_STEP,
        refine_stop_iter=cfg.DENSIFY_STOP_STEP,
        refine_every=cfg.DENSIFY_INTERVAL,
        grow_grad2d=cfg.DENSIFY_GRAD_THRESH,
        reset_every=cfg.OPACITY_RESET_INTERVAL,
    )
    strategy_state = strategy.initialize_state(scene_scale=cfg.SCENE_SCALE)

    # training loop
    train_aug_images = []
    pbar = tqdm(range(total_steps))
    for step in pbar:
        # gradually increase SH degree
        if step > 0 and step % cfg.SH_UPGRADE_INTERVAL == 0:
            model.sh_degree = min(model.sh_degree + 1, model.sh_degree_max)

        # sample random training image
        data = train_dataset[random.randint(0, num_train - 1)]

        gt_image = prepare_train_target(data["images"].to(device), cfg)  # (3, H, W)

        camtoworld = data["transforms"].to(device)  # (3, 4)
        H, W = gt_image.shape[1], gt_image.shape[2]

        # forward
        image_id = data["image_id"].to(device) if model.use_appearance else None
        rendered, alphas, info = model(camtoworld, H, W, image_id=image_id, canonical=False)

        # loss: (1 - lambda) * L1 + lambda * (1 - SSIM) + edge regularization
        gt_hwc = gt_image.permute(1, 2, 0)  # (H, W, 3)
        l1_loss = torch.abs(rendered - gt_hwc).mean()
        ssim_val = ssim(rendered, gt_hwc)
        recon_loss = (1.0 - cfg.LAMBDA_SSIM) * l1_loss + cfg.LAMBDA_SSIM * (1.0 - ssim_val)
        edge_lambda = float(getattr(cfg, "LAMBDA_EDGE", 0.0))
        edge_loss = image_gradient_l1(rendered, gt_hwc) if edge_lambda > 0 else torch.tensor(0.0, device=device)
        loss = recon_loss + edge_lambda * edge_loss

        # densification hooks
        strategy.step_pre_backward(model.splats, optimizers, strategy_state, step, info)
        loss.backward()
        strategy.step_post_backward(model.splats, optimizers, strategy_state, step, info, packed=False)

        # optimizer step
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sch in schedulers.values():
            sch.step()

        # logging
        if step % cfg.LOG_INTERVAL_STEP == 0:
            with torch.no_grad():
                mse = ((rendered - gt_hwc) ** 2).mean()
                psnr = -10.0 * math.log10(mse.clamp_min(1e-10).item())
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                psnr=f"{psnr:.2f}",
                edge=f"{edge_loss.item():.4f}",
                n_gs=model.num_gaussians,
            )

        # collect first training targets for visualization (only once)
        if train_aug_images is not None:
            train_aug_images.append(gt_image.clamp(0, 1))
            if len(train_aug_images) >= 4:
                grid = make_grid(train_aug_images[:4], nrow=2)
                save_image(grid, os.path.join(output_dir, "examples", "train_aug.jpg"))
                train_aug_images = None

        # validation
        if step > 0 and step % cfg.VAL_INTERVAL_STEP == 0:
            validate(model, val_dataset, step, device, output_dir, cfg)
            torch.save(model.state_dict(), os.path.join(output_dir, "latest.pt"))
            print(f"Model saved to {output_dir}/latest.pt")

    # save model checkpoint
    torch.save(model.state_dict(), os.path.join(output_dir, "latest.pt"))
    print(f"Model saved to {output_dir}/latest.pt")

    # run test evaluation
    test_dataset = Blender(meta_cfg.DATASET, split="test", load_images=False)
    evaluate(model, test_dataset, device, output_dir, cfg)


@torch.no_grad()
def validate(model, val_dataset, step, device, output_dir, model_cfg):
    model.eval()
    H, W = val_dataset._data_info["img_h"], val_dataset._data_info["img_w"]
    num_val = len(val_dataset._records_keys)
    val_images = []
    for i in range(num_val):
        data = val_dataset[i]
        camtoworld = data["transforms"].to(device)
        rendered, _, _ = model(camtoworld, H, W, canonical=True)
        rendered = lowlight_enhance(rendered, model_cfg)
        # collect first 4 rendered images for visual spot-checking
        if i < 4:
            val_images.append(rendered.permute(2, 0, 1).clamp(0, 1))
    # save 4 views as a 2x2 grid
    if val_images:
        grid = make_grid(val_images, nrow=2)
        save_image(grid, os.path.join(output_dir, "examples", f"val_step{step}.jpg"))
    print(f"\n[Step {step}] {model.num_gaussians} Gaussians")
    model.train()


@torch.no_grad()
def evaluate(model, test_dataset, device, output_dir, model_cfg):
    model.eval()
    H, W = test_dataset._data_info["img_h"], test_dataset._data_info["img_w"]
    num_test = len(test_dataset._records_keys)
    for i in range(num_test):
        data = test_dataset[i]
        camtoworld = data["transforms"].to(device)
        rendered, _, _ = model(camtoworld, H, W, canonical=True)
        rendered = lowlight_enhance(rendered, model_cfg)
        frame_name = test_dataset._records_keys[i]
        save_image(
            rendered.permute(2, 0, 1).clamp(0, 1),
            os.path.join(output_dir, "test", f"{frame_name}.png"),
        )
    print(f"Test renders saved to {output_dir}/test/")
    model.train()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", required=True, type=str)
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    train(args.config_path)
