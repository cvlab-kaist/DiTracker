# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import hydra
import numpy as np
import torch

from typing import Optional
from dataclasses import dataclass, field

from omegaconf import OmegaConf

from cotracker.datasets.utils import collate_fn
from cotracker.models.evaluation_predictor import EvaluationPredictor
from cotracker.utils.visualizer import Visualizer

from cotracker.evaluation.core.evaluator import Evaluator
from cotracker.models.build_cotracker import build_cotracker, build_dit_cotracker
from cotracker.datasets.tap_vid_datasets import CorruptedTapVidDataset


@dataclass(eq=False)
class DefaultConfig:
    # Directory where all outputs of the experiment will be saved.
    exp_dir: str = "./outputs"
    
    # Path to the pre-trained model checkpoint to be used for the evaluation.
    checkpoint: str = "/path/to/ckpt"
    
    # The root directory of the dataset.
    dataset_root: str = "/path/to/data"
    
    # Name of the dataset to be used for the evaluation.
    dataset_name: str = "tapvid_davis_first"
    
    # Severity of corruption
    severity: int = 5
    
    # Tracking head parameters
    model_stride: int = 4
    corr_radius: int = 3
    corr_levels: int = 4
    sliding_window_len: int = 60
    num_virtual_tracks: int = 64
    model_resolution: tuple = (480, 720)
    linear_layer_for_vis_conf: bool = True
    latent_dim: int = 64
    
    # DiTracker parameters
    model_path: str = "THUDM/CogVideoX-2B"
    layer_hook: int = 17
    head_hook: int = 2
    cost_softmax: bool = True
    resnet_fuse_mode: str = "concat"
    lora_rank: int = 128
    lora_alpha: int = 64
    
    # Visualizer parameters
    visualize: bool = False
    trace: int = 12
    linewidth: int = 5
    resize_to: tuple = (480, 720)
    
    # EvaluationPredictor parameters
    grid_size: int = 5
    local_grid_size: int = 8
    num_uniformly_sampled_pts: int = 0
    sift_size: int = 0
    
    # A flag indicating whether to evaluate one ground truth point at a time.
    single_point: bool = False
    offline_model: bool = True
    window_len: int = 60
    n_iters: int = 6

    seed: int = 0
    gpu_idx: int = 2
    local_extent: int = 50

    v2: bool = False

    # Override hydra's working directory to current working dir,
    # also disable storing the .hydra logs:
    hydra: dict = field(
        default_factory=lambda: {
            "run": {"dir": "."},
            "output_subdir": None,
        }
    )


def run_eval(cfg: DefaultConfig):
    """
    The function evaluates CoTracker on a specified benchmark dataset based on a provided configuration.

    Args:
        cfg (DefaultConfig): An instance of DefaultConfig class which includes:
            - exp_dir (str): The directory path for the experiment.
            - dataset_name (str): The name of the dataset to be used.
            - dataset_root (str): The root directory of the dataset.
            - checkpoint (str): The path to the CoTracker model's checkpoint.
            - single_point (bool): A flag indicating whether to evaluate one ground truth point at a time.
            - n_iters (int): The number of iterative updates for each sliding window.
            - seed (int): The seed for setting the random state for reproducibility.
            - gpu_idx (int): The index of the GPU to be used.
    """
    # Creating the experiment directory if it doesn't exist
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # Saving the experiment configuration to a .yaml file in the experiment directory
    cfg_file = os.path.join(cfg.exp_dir, "expconfig.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    evaluator = Evaluator(cfg.exp_dir)
    model = build_dit_cotracker(cfg)

    # Creating the EvaluationPredictor object
    predictor = EvaluationPredictor(
        model,
        grid_size=cfg.grid_size,
        local_grid_size=cfg.local_grid_size,
        sift_size=cfg.sift_size,
        single_point=cfg.single_point,
        num_uniformly_sampled_pts=cfg.num_uniformly_sampled_pts,
        n_iters=cfg.n_iters,
        local_extent=cfg.local_extent,
        interp_shape=cfg.model_resolution,
    )

    if torch.cuda.is_available():
        predictor.model = predictor.model.cuda()

    # Setting the random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Constructing the specified dataset
    curr_collate_fn = collate_fn
    if "tapvid" in cfg.dataset_name:
        dataset_type = cfg.dataset_name.split("_")[1]
        if dataset_type == "davis":
            data_root = os.path.join(
                cfg.dataset_root, "tapvid_davis", "tapvid_davis.pkl"
            )
        elif dataset_type == "kinetics":
            data_root = os.path.join(cfg.dataset_root, "tapvid_kinetics")
        elif dataset_type == "robotap":
            data_root = os.path.join(cfg.dataset_root, "tapvid_robotap")
        elif dataset_type == "stacking":
            data_root = os.path.join(
                cfg.dataset_root, "tapvid_rgb_stacking", "tapvid_rgb_stacking.pkl"
            )
    
    corruption_types = {
        'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
        'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
        'weather': ['snow', 'frost', 'fog', 'brightness'],
        'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    }
    all_results = {}
    for category, corruption_list in corruption_types.items():
        print(f"Evaluating {category.upper()} corruptions...")
        for corruption_name in corruption_list:
            exp_subdir = os.path.join(cfg.exp_dir, corruption_name)
            os.makedirs(exp_subdir, exist_ok=True)
            
            test_dataset = CorruptedTapVidDataset(
                dataset_type="davis",
                data_root=data_root,
                queried_first=True,
                corruption_name=corruption_name,
                corruption_severity=cfg.severity,
            )
            
            # Creating the DataLoader object
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                collate_fn=curr_collate_fn,
            )
            
            if cfg.visualize:
                visualizer = Visualizer(
                    save_dir=exp_subdir,
                    fps=7,
                    linewidth=cfg.linewidth,
                    tracks_leave_trace=cfg.trace,
                    resize_to=cfg.resize_to
                )
            else:
                visualizer = None

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    evaluate_result = evaluator.evaluate_sequence(
                        predictor, test_dataloader, dataset_name=cfg.dataset_name, visualizer=visualizer
                    )
                    evaluate_result = evaluate_result["avg"]
                    result_file = os.path.join(exp_subdir, "result_eval.json")
                    with open(result_file, "w") as f:
                        json.dump(evaluate_result, f, indent=2)
            all_results[corruption_name] = evaluate_result
            
    all_results_file = os.path.join(cfg.exp_dir, "all_results.json")
    with open(all_results_file, "w") as f:
        json.dump(all_results, f, indent=2)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="default_config_eval", node=DefaultConfig)


@hydra.main(config_path="./configs", config_name="default_config_eval")
def evaluate(cfg: DefaultConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    evaluate()
