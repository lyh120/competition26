#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import argparse
import os
import warnings

import torch
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from core.data import Blender
from core.libs import ConfigDict, lowlight_enhance
from core.model import Simple3DGS



@torch.no_grad()
def evaluate(checkpoint_path, device="cuda", canonical=True, appearance_image_id=0):
    # load config from checkpoint directory
    ckpt_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(config_path) as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    config_dict["EXP_STR"] = ""
    config_dict["TIME_STR"] = ""
    meta_cfg = ConfigDict(config_path=config_dict)
    cfg = meta_cfg.MODEL
    if bool(getattr(cfg, "USE_ADAPTIVE_RENDER_ENHANCE", True)):
        render_eps = float(getattr(cfg, "RENDER_EPS", 1e-6))
        target_mean = float(getattr(cfg, "RENDER_TARGET_MEAN", 0.5))
        max_scale = float(getattr(cfg, "RENDER_MAX_SCALE", 4.0))
        gamma_min = float(getattr(cfg, "RENDER_GAMMA_MIN", 0.4))
        gamma_max = float(getattr(cfg, "RENDER_GAMMA_MAX", 1.2))
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
        render_gamma = float(getattr(cfg, "RENDER_GAMMA", 1.0))
        if render_gamma <= 0:
            raise ValueError(f"RENDER_GAMMA must be > 0, got {render_gamma}.")

    # load dataset (json only, no images)
    test_dataset = Blender(meta_cfg.DATASET, split="test", load_images=False)
    H, W = test_dataset._data_info["img_h"], test_dataset._data_info["img_w"]

    # build model and load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    data_info = dict(test_dataset._data_info)
    if "appearance_embeddings.weight" in ckpt:
        data_info["num_train_images"] = ckpt["appearance_embeddings.weight"].shape[0]
    else:
        data_info["num_train_images"] = test_dataset._data_info["num_images"]
    model = Simple3DGS(cfg, data_info).to(device)
    model.load_state_dict(ckpt, strict=False)
    model.sh_degree = model.sh_degree_max
    model.eval()

    # render and save
    output_dir = os.path.join(ckpt_dir, "test")
    os.makedirs(output_dir, exist_ok=True)
    num_test = len(test_dataset._records_keys)
    use_appearance = (not canonical) and model.use_appearance
    if use_appearance:
        max_train_images = model.appearance_embeddings.num_embeddings
        if appearance_image_id < 0 or appearance_image_id >= max_train_images:
            raise ValueError(
                f"appearance_image_id must be in [0, {max_train_images - 1}], got {appearance_image_id}."
            )
        eval_image_id = torch.tensor(appearance_image_id, dtype=torch.long, device=device)
    else:
        eval_image_id = None

    for i in tqdm(range(num_test), desc="Rendering"):
        data = test_dataset[i]
        camtoworld = data["transforms"].to(device)
        rendered, _, _ = model(
            camtoworld,
            H,
            W,
            image_id=eval_image_id,
            canonical=canonical,
        )
        rendered = lowlight_enhance(rendered, cfg)
        frame_name = test_dataset._records_keys[i]
        save_image(
            rendered.permute(2, 0, 1).clamp(0, 1),
            os.path.join(output_dir, f"{frame_name}.png"),
        )
    mode = "canonical" if canonical else f"appearance(image_id={appearance_image_id})"
    print(f"Rendered {num_test} images to {output_dir}/ | {model.num_gaussians} Gaussians | mode={mode}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-w", required=True, type=str)
    parser.add_argument(
        "--canonical",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: disable appearance MLP (default), 0: enable appearance MLP with --appearance_image_id.",
    )
    parser.add_argument(
        "--appearance_image_id",
        type=int,
        default=0,
        help="Train image embedding index used when --canonical=0.",
    )
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        canonical=bool(args.canonical),
        appearance_image_id=args.appearance_image_id,
    )
