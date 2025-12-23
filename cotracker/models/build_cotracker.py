# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.ditracker import DiTracker


from peft import LoraConfig, set_peft_model_state_dict, get_peft_model
from diffusers import CogVideoXPipeline

def build_cotracker(checkpoint: str):
    cotracker = CoTrackerThreeOffline(
        stride=4, 
        corr_radius=3, 
        window_len=60, 
        model_resolution=(384, 512), 
        latent_dim=128, 
        resnet_fuse_mode=None
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        cotracker.load_state_dict(state_dict, strict=True)
    return cotracker



def build_dit_cotracker(cfg):
    model = DiTracker(
        stride=cfg.model_stride,
        corr_radius=cfg.corr_radius,
        corr_levels=cfg.corr_levels,
        window_len=cfg.sliding_window_len,
        num_virtual_tracks=cfg.num_virtual_tracks,
        model_resolution=cfg.model_resolution,
        linear_layer_for_vis_conf=cfg.linear_layer_for_vis_conf,
        latent_dim=cfg.latent_dim,
        
        model_path=cfg.model_path,
        layer_hooks=[cfg.layer_hook],
        head_hooks=[cfg.head_hook],
        cost_softmax=cfg.cost_softmax,
        resnet_fuse_mode=cfg.resnet_fuse_mode,
    )
    
    target_modules = []
    for target_layer_index in range(cfg.layer_hook + 1):
        if target_layer_index == cfg.layer_hook:
            sub_modules = ["to_q", "to_k"]
        else:
            sub_modules = ["to_q", "to_k", "to_v", "to_out.0"]
            
        target_modules.extend([
            name for name, _ in model.dit_fnet.transformer.named_modules()
            if f"blocks.{target_layer_index}." in name and any(sub in name for sub in sub_modules)
        ])
    
    transformer_lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        init_lora_weights=True,
        target_modules=sorted(set(target_modules)),
    )
    
    model.dit_fnet.transformer = get_peft_model(model.dit_fnet.transformer, transformer_lora_config)
    model._transformer_lora_cfg = transformer_lora_config

    model.dit_fnet.transformer.to(torch.bfloat16)
    model.dit_fnet.transformer.enable_gradient_checkpointing()
    
    lora_state_dict = CogVideoXPipeline.lora_state_dict(cfg.checkpoint)
    transformer_state_dict = {
        f'{k.replace("transformer.", "")}': v
        for k, v in lora_state_dict.items()
        if k.startswith("transformer.")
    }
    incompatible_keys = set_peft_model_state_dict(
        model.dit_fnet.transformer, transformer_state_dict, adapter_name="default"
    )
    model_state_dict = torch.load(os.path.join(cfg.checkpoint, "model.pth"), weights_only=False)
    model.load_state_dict(model_state_dict, strict=False)
        
    return model