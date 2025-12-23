# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import torch
import signal
import json
import torch.nn.functional as F
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
import ast

from torch.cuda.amp import GradScaler
from pytorch_lightning.lite import LightningLite

from cotracker.models.core.cotracker.ditracker import DiTracker

from cotracker.utils.visualizer import Visualizer
from cotracker.evaluation.core.evaluator import Evaluator
from cotracker.datasets.utils import collate_fn_train, dataclass_to_cuda_
from cotracker.models.core.cotracker.losses import (
    sequence_loss,
    sequence_BCE_loss,
    sequence_prob_loss,
)
from cotracker.utils.train_utils import (
    Logger,
    get_eval_dataloader,
    get_train_dataset,
    sig_handler,
    term_handler,
    run_test_eval,
)
import wandb
import torch.distributed as dist


from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict, get_peft_model
from diffusers import CogVideoXPipeline

def unwrap_module(m: torch.nn.Module) -> torch.nn.Module:
    # unwrap repeatedly in case there are multiple wrappers (DDP, precision, etc.)
    while hasattr(m, "module"):
        m = m.module
    return m


def fetch_optimizer(args, model):
    for attr_name, component in model.dit_fnet.named_modules():
        if hasattr(component, "requires_grad_"):
            if 'transformer.transformer_blocks.' in attr_name and int(attr_name.split('.')[2]) <= max(args.layer_hooks):
                component.requires_grad_(True)
            else:
                component.requires_grad_(False)
    
    target_modules = []
    for target_layer_index in range(max(args.layer_hooks) + 1):
        if target_layer_index == max(args.layer_hooks):
            sub_modules = ["to_q", "to_k"]
        else:
            sub_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        
        target_modules.extend([
            name for name, _ in model.dit_fnet.transformer.named_modules()
            if f"blocks.{target_layer_index}." in name and any(sub in name for sub in sub_modules)
        ])
        
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=sorted(set(target_modules)),
    )
    
    model.dit_fnet.transformer = get_peft_model(model.dit_fnet.transformer, transformer_lora_config)
    model._transformer_lora_cfg = transformer_lora_config

    model.dit_fnet.transformer.to(torch.bfloat16)
    model.dit_fnet.transformer.enable_gradient_checkpointing()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params / (1024 ** 2):.2f}M")
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="cos",
    )
    return optimizer, scheduler


def forward_batch(batch, model, args):
    video = batch.video
    trajs_g = batch.trajectory
    vis_g = batch.visibility
    valids = batch.valid

    B, T, C, H, W = video.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = video.device

    __, first_positive_inds = torch.max(vis_g, dim=1)

    if args.query_sampling_method == "random":
        assert B == 1
        true_indices = torch.nonzero(vis_g[0])
        # Group the indices by the first column (N)
        grouped_indices = true_indices[:, 1].unique()
        # Initialize an empty tensor to hold the sampled points
        sampled_points = torch.empty((B, N, D))
        indices = torch.empty((B, N, 1))
        # For each unique N
        for n in grouped_indices:
            # Get the T indices where visibilities[0, :, n] is True
            t_indices = true_indices[true_indices[:, 1] == n, 0]

            # Select a random index from t_indices
            random_index = t_indices[torch.randint(0, len(t_indices), (1,))]

            # Use this random index to sample a point from the trajectories tensor
            sampled_points[0, n] = trajs_g[0, random_index, n]
            indices[0, n] = random_index.float()
        # model.window_len = vis_g.shape[1]
        queries = torch.cat([indices, sampled_points], dim=2)
    else:
        # We want to make sure that during training the model sees visible points
        # that it does not need to track just yet: they are visible but queried from a later frame
        N_rand = N // 4
        # inds of visible points in the 1st frame
        nonzero_inds = [
            [torch.nonzero(vis_g[b, :, i]) for i in range(N)] for b in range(B)
        ]

        for b in range(B):
            rand_vis_inds = torch.cat(
                [
                    nonzero_row[torch.randint(len(nonzero_row), size=(1,))]
                    for nonzero_row in nonzero_inds[b]
                ],
                dim=1,
            )
            first_positive_inds[b] = torch.cat(
                [rand_vis_inds[:, :N_rand], first_positive_inds[b : b + 1, N_rand:]],
                dim=1,
            )

        ind_array_ = torch.arange(T, device=device)
        ind_array_ = ind_array_[None, :, None].repeat(B, 1, N)
        assert torch.allclose(
            vis_g[ind_array_ == first_positive_inds[:, None, :]],
            torch.ones(1, device=device),
        )
        gather = torch.gather(
            trajs_g, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, D)
        )
        xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)

        queries = torch.cat([first_positive_inds[:, :, None], xys[:, :, :2]], dim=2)

    assert B == 1

    if (
        torch.isnan(queries).any()
        or torch.isnan(trajs_g).any()
        or queries.abs().max() > 1500
    ):
        print("failed_sample")
        print("queries time", queries[..., 0])
        print("queries ", queries[..., 1:])
        queries = torch.ones_like(queries).to(queries.device).float()
        print("new queries", queries)
        valids = torch.zeros_like(valids).to(valids.device).float()
        print("new valids", valids)

    model_output = model(
        video=video, queries=queries[..., :3], iters=args.train_iters, is_train=True
    )

    tracks, visibility, confidence, train_data = model_output
    coord_predictions, vis_predictions, confidence_predicitons, valid_mask = train_data

    vis_gts = []
    invis_gts = []
    traj_gts = []
    valids_gts = []

    S = T
    seq_len = (S // 2) + 1

    for ind in range(0, seq_len - S // 2, S // 2):
        vis_gts.append(vis_g[:, ind : ind + S])
        invis_gts.append(1 - vis_g[:, ind : ind + S])
        traj_gts.append(trajs_g[:, ind : ind + S, :, :2])
        val = valids[:, ind : ind + S]
        valids_gts.append(val)

    seq_loss_visible = sequence_loss(
        coord_predictions,
        traj_gts,
        valids_gts,
        vis=vis_gts,
        gamma=0.8,
        add_huber_loss=args.add_huber_loss,
        loss_only_for_visible=True,
    )
    confidence_loss = sequence_prob_loss(
        coord_predictions, confidence_predicitons, traj_gts, vis_gts
    )
    vis_loss = sequence_BCE_loss(vis_predictions, vis_gts)

    output = {"flow": {"predictions": tracks[0].detach()}}
    output["flow"]["loss"] = seq_loss_visible.mean() * 0.05
    output["flow"]["queries"] = queries.clone()

    if not args.train_only_on_visible:
        seq_loss_invisible = sequence_loss(
            coord_predictions,
            traj_gts,
            valids_gts,
            vis=invis_gts,
            gamma=0.8,
            add_huber_loss=False,
            loss_only_for_visible=True,
        )
        output["flow_invisible"] = {"loss": seq_loss_invisible.mean() * 0.01}
    output["visibility"] = {
        "loss": vis_loss.mean(),
        "predictions": visibility[0].detach(),
    }
    output["confidence"] = {
        "loss": confidence_loss.mean(),
    }
    return output


class Lite(LightningLite):
    def run(self, args):
        def seed_everything(seed: int):
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed_everything(42)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed + worker_id)
            random.seed(worker_seed + worker_id)

        g = torch.Generator()
        g.manual_seed(42)
        if self.global_rank == 0:
            eval_dataloaders = []
            for ds_name in args.eval_datasets:
                eval_dataloaders.append(
                    (ds_name, get_eval_dataloader(args.dataset_root, ds_name))
                )
            if not args.debug:
                final_dataloaders = [dl for dl in eval_dataloaders]

                ds_name = "dynamic_replica"
                final_dataloaders.append(
                    (ds_name, get_eval_dataloader(args.dataset_root, ds_name))
                )

                ds_name = "tapvid_robotap"
                final_dataloaders.append(
                    (ds_name, get_eval_dataloader(args.dataset_root, ds_name))
                )

                ds_name = "tapvid_kinetics_first"
                final_dataloaders.append(
                    (ds_name, get_eval_dataloader(args.dataset_root, ds_name))
                )

            evaluator = Evaluator(args.ckpt_path)

            visualizer = Visualizer(
                save_dir=args.ckpt_path,
                pad_value=180,
                fps=7,
                show_first_frame=0,
                tracks_leave_trace=0,
            )
        
        model = DiTracker(
            # CoTracker3 head params
            stride=args.model_stride,
            corr_radius=args.corr_radius,
            corr_levels=args.corr_levels,
            window_len=args.sliding_window_len,
            num_virtual_tracks=args.num_virtual_tracks,
            model_resolution=args.model_resolution,
            linear_layer_for_vis_conf=args.linear_layer_for_vis_conf,
            latent_dim=args.latent_dim,
            
            # DiT backbone params
            model_path=args.model_path,
            layer_hooks=args.layer_hooks,
            head_hooks=args.head_hooks,
            cost_softmax=args.cost_softmax,
            resnet_fuse_mode=args.resnet_fuse_mode,
        )
        with open(args.ckpt_path + "/meta.json", "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=4)

        model.cuda()

        train_dataset = get_train_dataset(args)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
            collate_fn=collate_fn_train,
            drop_last=True,
        )
        train_loader = self.setup_dataloaders(train_loader, move_to_device=False)
        print("LEN TRAIN LOADER", len(train_loader))
        optimizer, scheduler = fetch_optimizer(args, model)

        total_steps = 0
        if self.global_rank == 0:
            logger = Logger(model, scheduler, ckpt_path=args.ckpt_path)
            wandb.init(
                project="DiTracker",
                entity="sonsoowon",
                name=f"{args.exp_name}",
                config=vars(args),
                dir=args.ckpt_path,
            )

        # Resume from checkpoint
        total_steps = 0
        folder_ckpts = [f for f in os.listdir(args.ckpt_path) if "iter" in f]
        if len(folder_ckpts) > 0:
            resume_dir = folder_ckpts[-1]
            logging.info(f"Loading checkpoint {resume_dir}")
            
            real_model = unwrap_module(model)
            
            lora_state_dict = CogVideoXPipeline.lora_state_dict(resume_dir)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            set_peft_model_state_dict(
                real_model.dit_fnet.transformer, transformer_state_dict, adapter_name="default"
            )
            del lora_state_dict, transformer_state_dict
            logging.info("Load DiT LoRA")
            
            model_state_dict = torch.load(
                os.path.join(resume_dir, "model.pth"), 
                map_location='cpu',
                weights_only=False
            )
            real_model.load_state_dict(model_state_dict, strict=False)
            del model_state_dict
            logging.info("Load feat upsampler")
            
            training_state = self.load(
                os.path.join(resume_dir, "train_state.pth"),
            )
            if "optimizer" in training_state:
                optimizer.load_state_dict(training_state["optimizer"])
            if "scheduler" in training_state:
                scheduler.load_state_dict(training_state["scheduler"])
            if "total_steps" in training_state:
                total_steps = training_state["total_steps"]
            del training_state
            
            torch.cuda.empty_cache()
            
        model, optimizer = self.setup(model, optimizer, move_to_device=False)    
        
        model.train()

        save_freq = args.save_freq
        scaler = GradScaler(enabled=False)

        should_keep_training = True
        global_batch_num = 0
        epoch = -1

        while should_keep_training:
            epoch += 1
            for i_batch, batch in enumerate(tqdm(train_loader)):
                batch, gotit = batch
                local_ok = torch.tensor(1 if all(gotit) else 0, device="cuda")
                
                if dist.is_initialized():
                    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
                            
                if local_ok.item() == 0:
                    print("batch is None")
                    continue

                dataclass_to_cuda_(batch)

                optimizer.zero_grad(set_to_none=True)

                assert model.training

                output = forward_batch(batch, model, args)

                loss = 0
                for k, v in output.items():
                    if "loss" in v:
                        loss += v["loss"]

                if self.global_rank == 0:
                    for k, v in output.items():
                        if "loss" in v:
                            logger.writer.add_scalar(
                                f"live_{k}_loss", v["loss"].item(), total_steps
                            )
                        if "metrics" in v:
                            logger.push(v["metrics"], k)
                        wandb.log({f"train/{k}_loss": v["loss"].item()}, step=total_steps)
                    wandb.log({"train/loss": loss.item()}, step=total_steps)
                    
                    if total_steps % save_freq == save_freq - 1:
                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=batch.trajectory.clone()[..., :2],
                            visibility=batch.visibility.clone(),
                            filename="train_gt_traj_0",
                            writer=None,
                            step=total_steps,
                        )

                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=output["flow"]["predictions"][None],
                            visibility=output["visibility"]["predictions"][None] > 0.8,
                            filename="train_pred_traj_0",
                            writer=None,
                            step=total_steps,
                        )
                        
                        wandb.log({"vis/gt_video": wandb.Video(os.path.join(args.ckpt_path, "train_gt_traj_0.mp4"), fps=10, format="mp4")}, step=total_steps)
                        wandb.log({"vis/pred_video": wandb.Video(os.path.join(args.ckpt_path, "train_pred_traj_0.mp4"), fps=10, format="mp4")}, step=total_steps)

                    if len(output) > 1:
                        logger.writer.add_scalar(
                            f"live_total_loss", loss.item(), total_steps
                        )
                    logger.writer.add_scalar(
                        f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
                    )
                    wandb.log({"train/lr": optimizer.param_groups[0]["lr"]}, step=total_steps)
                    
                    global_batch_num += 1

                self.backward(scaler.scale(loss))

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                total_steps += 1
                
                del batch, output, loss
                torch.cuda.empty_cache()
                
                if (i_batch >= len(train_loader) - 1) or (
                    total_steps == 1 and args.validate_at_start
                ):
                    if dist.is_initialized(): dist.barrier()
                    if self.global_rank == 0:
                        if (epoch + 1) % args.save_every_n_epoch == 0:
                            ckpt_iter = f"{total_steps:06d}"
                            save_dir = f"{args.ckpt_path}/iter{ckpt_iter}"
                            os.makedirs(save_dir, exist_ok=True)
                            real_model = unwrap_module(model)
                            
                            lora_state_dict = get_peft_model_state_dict(real_model.dit_fnet.transformer)
                            CogVideoXPipeline.save_lora_weights(save_dir, lora_state_dict)
                            
                            model_state_dict = {
                                k: v for k, v in real_model.state_dict().items() 
                                if "dit_fnet" not in k
                            }
                            torch.save(model_state_dict, os.path.join(save_dir, "model.pth"))

                            train_state_dict = {
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "total_steps": total_steps,
                            }
                            logging.info(f"Saving file {save_dir}")
                            self.save(train_state_dict, os.path.join(save_dir, "train_state.pth"))

                        if (epoch + 1) % args.evaluate_every_n_epoch == 0 or (
                            args.validate_at_start and epoch == 0
                        ):
                            eval_model = unwrap_module(model)
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                with torch.no_grad():
                                    metrics = run_test_eval(
                                        evaluator,
                                        eval_model,
                                        eval_dataloaders,
                                        logger.writer,
                                        total_steps,
                                        query_random=(
                                            args.query_sampling_method is not None
                                            and "random" in args.query_sampling_method
                                        ),
                                        visualizer=visualizer,
                                        interp_shape=args.crop_size,
                                    )
                            for metric in metrics:
                                wandb.log({"eval_metrics": metric}, step=total_steps)
                            torch.cuda.empty_cache()
                    if dist.is_initialized(): dist.barrier()
                    model.train()
                elif (
                    total_steps % args.evaluate_every_n_step == 0
                    or total_steps % args.save_every_n_step == 0
                ):
                    if dist.is_initialized(): dist.barrier()
                    if self.global_rank == 0:
                        if total_steps % args.evaluate_every_n_step == 0:
                            eval_model = unwrap_module(model)
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                with torch.no_grad():
                                    metrics = run_test_eval(
                                        evaluator,
                                        eval_model,
                                        eval_dataloaders,
                                        logger.writer,
                                        total_steps,
                                        query_random=(
                                            args.query_sampling_method is not None
                                            and "random" in args.query_sampling_method
                                        ),
                                        visualizer=visualizer,
                                        interp_shape=args.crop_size,
                                    )
                            for metric in metrics:
                                wandb.log({"eval_metrics": metric}, step=total_steps)
                            
                            torch.cuda.empty_cache()
                        
                        if total_steps % args.save_every_n_step == 0:
                            ckpt_iter = f"{total_steps:06d}"
                            save_dir = f"{args.ckpt_path}/iter{ckpt_iter}"
                            os.makedirs(save_dir, exist_ok=True)
                            real_model = unwrap_module(model)
                            
                            lora_state_dict = get_peft_model_state_dict(real_model.dit_fnet.transformer)
                            CogVideoXPipeline.save_lora_weights(save_dir, lora_state_dict)
                            
                            # Extract feat_upsampler parameters (full, not LoRA)
                            model_state_dict = {
                                k: v for k, v in real_model.state_dict().items() 
                                if "dit_fnet" not in k
                            }
                            torch.save(model_state_dict, os.path.join(save_dir, "model.pth"))
                                
                            train_state_dict = {
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "total_steps": total_steps,
                            }
                            logging.info(f"Saving file {save_dir}")
                            self.save(train_state_dict, os.path.join(save_dir, "train_state.pth"))
                    if dist.is_initialized(): dist.barrier()
                    model.train()
                if total_steps > args.num_steps:
                    should_keep_training = False
                    break

        if self.global_rank == 0:
            print("FINISHED TRAINING")
            save_dir = f"{args.ckpt_path}/iter{ckpt_iter}_final"
            real_model = unwrap_module(model)
            
            lora_state_dict = get_peft_model_state_dict(real_model.dit_fnet.transformer)
            CogVideoXPipeline.save_lora_weights(save_dir, lora_state_dict)
            
            # Extract feat_upsampler parameters (full, not LoRA)
            model_state_dict = {
                k: v for k, v in real_model.state_dict().items() 
                if "dit_fnet" not in k
            }
            torch.save(model_state_dict, os.path.join(save_dir, "model.pth"))

            run_test_eval(
                evaluator,
                model,
                final_dataloaders,
                logger.writer,
                total_steps,
                query_random=(
                    args.query_sampling_method is not None
                    and "random" in args.query_sampling_method
                ),
                visualizer=visualizer,
                interp_shape=args.crop_size,
            )
            logger.close()


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ckpt_path", help="path to save checkpoints")
    parser.add_argument("--exp_name", type=str, default="train_ditracker")
    
    ## DiT parameters ##
    parser.add_argument("--model_path", default="THUDM/CogVideoX-2B")
    parser.add_argument("--layer_hooks", type=int, nargs='+', default=[17])
    parser.add_argument("--head_hooks", type=str, nargs='+', default=[2])
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--cost_softmax", type=bool, default=True)
    parser.add_argument("--resnet_fuse_mode", type=str, default="concat")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=64)
    
    ## Tracking head parameters ##
    parser.add_argument("--train_iters", type=int, default=4)
    parser.add_argument("--num_virtual_tracks", type=int, default=64)
    parser.add_argument("--sliding_window_len", type=int, default=60)
    parser.add_argument("--model_stride", type=int, default=4)
    parser.add_argument("--corr_radius", type=int, default=3)
    parser.add_argument("--corr_levels", type=int, default=4)
    parser.add_argument("--linear_layer_for_vis_conf", type=bool, default=True)
    parser.add_argument(
        "--model_resolution",
        type=int,
        nargs="+",
        default=[480, 720],
        help="crop videos to this resolution during training",
    )

    ## Dataset parameters ##
    parser.add_argument("--dataset_root", type=str, default="/path/to/data")
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs="+",
        default=[480, 720],
        help="crop videos to this resolution during training",
    )
    parser.add_argument("--eval_datasets", nargs="+", default=["tapvid_davis_first"])
    parser.add_argument("--train_datasets", nargs="+", default=["kubric"])
    parser.add_argument("--traj_per_sample", type=int, default=512)
    parser.add_argument("--sequence_len", type=int, default=46)
    parser.add_argument("--dont_use_augs", type=bool, default=False)
    parser.add_argument("--random_frame_rate", type=bool, default=True)
    parser.add_argument("--random_number_traj", type=bool, default=False)
    parser.add_argument("--random_seq_len", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=10)
    
    parser.add_argument("--query_sampling_method", type=str, default="random")
    parser.add_argument("--eval_max_seq_len", type=int, default=1000)
    
    
    ## Training parameters ## 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.0005, help="max learning rate.")
    parser.add_argument("--wdecay", type=float, default=0.0005, help="Weight decay in optimizer.")
    parser.add_argument("--add_huber_loss", type=bool, default=True)
    parser.add_argument("--train_only_on_visible", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=True)
    
    
    ## Logging parameters ##
    parser.add_argument(
        "--validate_at_start",
        action="store_true",
        help="whether to run evaluation before training starts",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="frequency of trajectory visualization during training",
    )
    parser.add_argument("--evaluate_every_n_step", type=int, default=1000)
    parser.add_argument("--save_every_n_step", type=int, default=1000)
    parser.add_argument(
        "--evaluate_every_n_epoch",
        type=int,
        default=3,
        help="evaluate during training after every n epochs, after every epoch by default",
    )
    parser.add_argument(
        "--save_every_n_epoch",
        type=int,
        default=3,
        help="save checkpoints during training after every n epochs, after every epoch by default",
    )

    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)
    from pytorch_lightning.strategies import DDPStrategy

    Lite(
        strategy=DDPStrategy(find_unused_parameters=False),
        devices="auto",
        accelerator="gpu",
        precision="bf16" if args.mixed_precision else 32,
        num_nodes=args.num_nodes,
    ).run(args)
