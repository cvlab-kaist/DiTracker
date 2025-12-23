# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import os
from typing import Optional
import torch
from tqdm import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from cotracker.datasets.utils import dataclass_to_cuda_
from cotracker.utils.visualizer import Visualizer
from cotracker.models.core.model_utils import reduce_masked_mean
from cotracker.evaluation.core.eval_utils import compute_tapvid_metrics
from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline
import logging
import wandb

from cotracker.utils.visualizer import concat_variant_across_i

class Evaluator:
    """
    A class defining the CoTracker evaluator.
    """

    def __init__(self, exp_dir, wandb_log=True) -> None:
        # Visualization
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        self.visualization_filepaths = defaultdict(lambda: defaultdict(list))
        self.visualize_dir = os.path.join(exp_dir, "visualisations")
        self.wandb_log = wandb_log
        
        self.all_OA_list = defaultdict(list)
        self.all_delta_avg_list = defaultdict(list)
        self.all_AJ_list = defaultdict(list)
        

    def compute_metrics(self, metrics, sample, pred_trajectory, dataset_name, ind=0):
        if isinstance(pred_trajectory, tuple):
            pred_trajectory, pred_visibility = pred_trajectory
        else:
            pred_visibility = None
        if "tapvid" in dataset_name:
            B, T, N, D = sample.trajectory.shape
            traj = sample.trajectory.clone()
            thr = 0.6

            if pred_visibility is None:
                logging.warning("visibility is NONE")
                pred_visibility = torch.zeros_like(sample.visibility)

            if not pred_visibility.dtype == torch.bool:
                pred_visibility = pred_visibility > thr

            query_points = sample.query_points.clone().cpu().numpy()

            pred_visibility = pred_visibility[:, :, :N]
            pred_trajectory = pred_trajectory[:, :, :N]

            gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
            gt_occluded = (
                torch.logical_not(sample.visibility.clone().permute(0, 2, 1))
                .cpu()
                .numpy()
            )

            pred_occluded = (
                torch.logical_not(pred_visibility.clone().permute(0, 2, 1))
                .cpu()
                .numpy()
            )
            pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()
            
            out_metrics = compute_tapvid_metrics(
                query_points,
                gt_occluded,
                gt_tracks,
                pred_occluded,
                pred_tracks,
                query_mode="strided" if "strided" in dataset_name else "first",
            )

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = np.mean(
                    [v[metric_name] for k, v in metrics.items() if k != "avg"]
                )

            logging.info(f"Metrics: {out_metrics}")
            logging.info(f"avg: {metrics['avg']}")
            print("metrics", out_metrics)
            print("avg", metrics["avg"])
        elif "itto" in dataset_name:
            B, T, N, D = sample.trajectory.shape
            traj = sample.trajectory.clone()
            thr = 0.6

            if pred_visibility is None:
                logging.warning("visibility is NONE")
                pred_visibility = torch.zeros_like(sample.visibility)

            if not pred_visibility.dtype == torch.bool:
                pred_visibility = pred_visibility > thr

            query_points = sample.query_points.clone().cpu().numpy()

            pred_visibility = pred_visibility[:, :, :N]
            pred_trajectory = pred_trajectory[:, :, :N]

            gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
            gt_occluded = (
                torch.logical_not(sample.visibility.clone().permute(0, 2, 1))
                .cpu()
                .numpy()
            )

            pred_occluded = (
                torch.logical_not(pred_visibility.clone().permute(0, 2, 1))
                .cpu()
                .numpy()
            )
            pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()
            
            
            # if "itto" in dataset_name:
            # motion dynamics
            disp = gt_tracks[:, :, 1:] - gt_tracks[:, :, :-1]
            disp = (np.sqrt(disp[..., 0] ** 2 + disp[..., 1] ** 2) / (np.sqrt(2) * 256))
            both_visible = (~gt_occluded[..., :-1]) & (~gt_occluded[..., 1:])
            
            disp_masked_sum = np.sum(disp * both_visible, axis=2)
            valid_counts = np.sum(both_visible, axis=2)
            avg_disp = disp_masked_sum / np.clip(valid_counts, a_min=1, a_max=None)
            
            motion_0_mask = avg_disp < 0.005
            motion_1_mask = (avg_disp >= 0.005) & (avg_disp < 0.015)
            motion_2_mask = (avg_disp >= 0.015) & (avg_disp < 0.05)
            motion_3_mask = (avg_disp >= 0.05)

            def _indices_per_batch(mask_bn: torch.Tensor):
                # returns a list of length B; each item is a 1D LongTensor of track indices in [0..N-1]
                out = []
                for b in range(mask_bn.shape[0]):
                    out.append(np.where(mask_bn[b])[0])
                return out
            
            motion_groups = {
                "motion0_[0,0.005)": _indices_per_batch(motion_0_mask),
                "motion1_[0.005,0.015)": _indices_per_batch(motion_1_mask),
                "motion2_[0.015,0.05)": _indices_per_batch(motion_2_mask),
                "motion3_[0.05,1.0)": _indices_per_batch(motion_3_mask),
            }

            # reappearance
            reappear_mask = gt_occluded[..., :-1] & (~gt_occluded[..., 1:])
            reappear_counts = np.sum(reappear_mask, axis=2)
            rep_0_mask = (reappear_counts == 0)
            rep_1_mask = (reappear_counts >= 1) & (reappear_counts <= 2)
            rep_2_mask = (reappear_counts >= 3)

            reappearance_groups = {
                "occ0_[0,1)": _indices_per_batch(rep_0_mask),
                "occ1_[1,3)": _indices_per_batch(rep_1_mask),
                "occ2_[3,1000)": _indices_per_batch(rep_2_mask),
            }
            
            OA_list, delta_avg_list, AJ_list = [], [], []
            for track_id in range(N):
                out_metrics = compute_tapvid_metrics(
                    query_points=query_points[:, track_id : track_id + 1, :],
                    gt_occluded=gt_occluded[:, track_id : track_id + 1, :],
                    gt_tracks=gt_tracks[:, track_id : track_id + 1, :],
                    pred_occluded=pred_occluded[:, track_id : track_id + 1, :],
                    pred_tracks=pred_tracks[:, track_id : track_id + 1, :],
                    query_mode="first",
                )

                OA_list.append(out_metrics["occlusion_accuracy"])
                delta_avg_list.append(out_metrics["average_pts_within_thresh"])
                AJ_list.append(out_metrics["average_jaccard"])
            
            for k, indices in motion_groups.items():
                self.all_OA_list[k].extend([OA_list[i] for i in indices[0].tolist()])
                self.all_delta_avg_list[k].extend([delta_avg_list[i] for i in indices[0].tolist()])
                self.all_AJ_list[k].extend([AJ_list[i] for i in indices[0].tolist()])
                
            for k, indices in reappearance_groups.items():
                self.all_OA_list[k].extend([OA_list[i] for i in indices[0].tolist()])
                self.all_delta_avg_list[k].extend([delta_avg_list[i] for i in indices[0].tolist()])
                self.all_AJ_list[k].extend([AJ_list[i] for i in indices[0].tolist()])
            
            out_metrics = compute_tapvid_metrics(
                query_points,
                gt_occluded,
                gt_tracks,
                pred_occluded,
                pred_tracks,
                query_mode="strided" if "strided" in dataset_name else "first",
            )

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = np.mean(
                    [v[metric_name] for k, v in metrics.items() if k != "avg"]
                )

            logging.info(f"Metrics: {out_metrics}")
            logging.info(f"avg: {metrics['avg']}")
            print("metrics", out_metrics)
            print("avg", metrics["avg"])
    
        elif dataset_name == "dynamic_replica" or dataset_name == "pointodyssey":
            *_, N, _ = sample.trajectory.shape
            B, T, N = sample.visibility.shape
            H, W = sample.video.shape[-2:]
            device = sample.video.device

            out_metrics = {}

            d_vis_sum = d_occ_sum = d_sum_all = 0.0
            thrs = [1, 2, 4, 8, 16]
            sx_ = (W - 1) / 255.0
            sy_ = (H - 1) / 255.0
            sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
            sc_pt = torch.from_numpy(sc_py).float().to(device)
            __, first_visible_inds = torch.max(sample.visibility, dim=1)

            frame_ids_tensor = torch.arange(T, device=device)[None, :, None].repeat(
                B, 1, N
            )
            start_tracking_mask = frame_ids_tensor > (first_visible_inds.unsqueeze(1))

            for thr in thrs:
                d_ = (
                    torch.norm(
                        pred_trajectory[..., :2] / sc_pt
                        - sample.trajectory[..., :2] / sc_pt,
                        dim=-1,
                    )
                    < thr
                ).float()  # B,S-1,N
                d_occ = (
                    reduce_masked_mean(
                        d_, (1 - sample.visibility) * start_tracking_mask
                    ).item()
                    * 100.0
                )
                d_occ_sum += d_occ
                out_metrics[f"accuracy_occ_{thr}"] = d_occ

                d_vis = (
                    reduce_masked_mean(
                        d_, sample.visibility * start_tracking_mask
                    ).item()
                    * 100.0
                )
                d_vis_sum += d_vis
                out_metrics[f"accuracy_vis_{thr}"] = d_vis

                d_all = reduce_masked_mean(d_, start_tracking_mask).item() * 100.0
                d_sum_all += d_all
                out_metrics[f"accuracy_{thr}"] = d_all

            d_occ_avg = d_occ_sum / len(thrs)
            d_vis_avg = d_vis_sum / len(thrs)
            d_all_avg = d_sum_all / len(thrs)

            sur_thr = 50
            dists = torch.norm(
                pred_trajectory[..., :2] / sc_pt - sample.trajectory[..., :2] / sc_pt,
                dim=-1,
            )  # B,S,N
            dist_ok = 1 - (dists > sur_thr).float() * sample.visibility  # B,S,N
            survival = torch.cumprod(dist_ok, dim=1)  # B,S,N
            out_metrics["survival"] = torch.mean(survival).item() * 100.0

            out_metrics["accuracy_occ"] = d_occ_avg
            out_metrics["accuracy_vis"] = d_vis_avg
            out_metrics["accuracy"] = d_all_avg

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = float(
                    np.mean([v[metric_name] for k, v in metrics.items() if k != "avg"])
                )

            logging.info(f"Metrics: {out_metrics}")
            logging.info(f"avg: {metrics['avg']}")
            print("metrics", out_metrics)
            print("avg", metrics["avg"])

    @torch.no_grad()
    def evaluate_sequence(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        train_mode=False,
        visualizer: Optional[Visualizer] = None,
    ):
        metrics = {}
        for ind, sample in enumerate(tqdm(test_dataloader)):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue
            if torch.cuda.is_available():
                dataclass_to_cuda_(sample)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            if (
                not train_mode
                and hasattr(model, "sequence_len")
                and (sample.visibility[:, : model.sequence_len].sum() == 0)
            ):
                print(f"skipping batch {ind}")
                continue

            if "tapvid" in dataset_name:
                queries = sample.query_points.clone().float()

                queries = torch.stack(
                    [
                        queries[:, :, 0],
                        queries[:, :, 2],
                        queries[:, :, 1],
                    ],
                    dim=2,
                ).to(device)
            elif "itto" in dataset_name:
                queries = sample.query_points.clone().float().to(device)
            else:
                queries = torch.cat(
                    [
                        torch.zeros_like(sample.trajectory[:, 0, :, :1]),
                        sample.trajectory[:, 0],
                    ],
                    dim=2,
                ).to(device)

            pred_tracks = model(sample.video, queries)

            if "strided" in dataset_name:
                inv_video = sample.video.flip(1).clone()
                inv_queries = queries.clone()
                inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

                pred_trj, pred_vsb = pred_tracks
                inv_pred_trj, inv_pred_vsb = model(inv_video, inv_queries)

                inv_pred_trj = inv_pred_trj.flip(1)
                inv_pred_vsb = inv_pred_vsb.flip(1)

                mask = pred_trj == 0

                pred_trj[mask] = inv_pred_trj[mask]
                pred_vsb[mask[:, :, :, 0]] = inv_pred_vsb[mask[:, :, :, 0]]

                pred_tracks = pred_trj, pred_vsb

            if dataset_name == "badja" or dataset_name == "fastcapture":
                seq_name = sample.seq_name[0]
            else:
                seq_name = str(ind)
            # breakpoint()
            if visualizer is not None:
                
                pred_visibility = pred_tracks[1] > 0.5
                gt_visibility = sample.visibility > 0.5
                visualizer.visualize(
                    video=sample.video.clone(),  # B, T, 3, H, W
                    tracks=pred_tracks[0].clone(),  # B, T, N 2
                    visibility=pred_visibility.clone(),  # B, T, N
                    filename=f"{seq_name}_pred"
                )
                visualizer.visualize(
                    video=sample.video.clone(),  # B, T, 3, H, W
                    tracks=sample.trajectory.clone(),  # B, T, N 2
                    visibility=gt_visibility.clone(),  # B, T, N
                    filename=f"{seq_name}_gt"
                )
            
            self.compute_metrics(metrics, sample, pred_tracks, dataset_name, ind)
        
        if "itto" in dataset_name:
            for k, v in self.all_AJ_list.items():
                metrics["avg"][f"{k}_AJ"] = np.mean(v)
            for k, v in self.all_OA_list.items():
                metrics["avg"][f"{k}_OA"] = np.mean(v)
            for k, v in self.all_delta_avg_list.items():
                metrics["avg"][f"{k}_delta_avg"] = np.mean(v)
                
        return metrics
