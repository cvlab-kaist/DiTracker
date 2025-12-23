import os
import torch
import numpy as np
from typing import List
from PIL import Image

import mediapy as media
from typing import Tuple

from cotracker.datasets.utils import CoTrackerData

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from utils import load_mose_video
def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
  """Resize a video to output_size."""
  # If you have a GPU, consider replacing this with a GPU-enabled resize op,
  # such as a jitted jax.image.resize.  It will make things faster.
  return media.resize_video(video, output_size)



# eek redundant... TODO(ilona) figure out importing of from utils here
def load_mose_video(video_folder):
    files = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    frames = []

    for filename in files:
        filepath = os.path.join(video_folder, filename)
        # Open image and ensure it's in RGB mode
        img = Image.open(filepath).convert("RGB")
        # Convert the image to a NumPy array of shape (H, W, 3)
        img_np = np.array(img)
        # Convert to (3, H, W) by reordering the axes
        # img_np = np.transpose(img_np, (2, 0, 1))
        frames.append(img_np)

    video_np = np.stack(frames, axis=0)
    # video_tensor = torch.from_numpy(video_np).float()

    # [T, 3, H, W]
    return video_np


class MoseTrackDataLoader:
    def __init__(
        self,
        video_ids: List[str],
        annotation_dir: str,
        video_dir: str,
        device: str = "cuda:0",
    ):
        self.video_ids = video_ids
        self.annotation_dir = annotation_dir
        self.video_dir = video_dir
        self.device = torch.device(device)

    def _load_annotations(self, file_path: str) -> torch.Tensor:
        return torch.tensor(
            np.load(file_path), dtype=torch.float32
        )  # [N, T, 4]

    def _create_queries(self, tracks: torch.Tensor) -> torch.Tensor:
        B, N, T, _ = tracks.shape
        query_coords = tracks[:, :, 0]  # [B, N, 2]
        query_frames = torch.zeros(B, N, device=self.device)  # [B, N]

        return torch.stack(
            [query_frames, query_coords[:, :, 0], query_coords[:, :, 1]], dim=2
        )

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        
        video_id = self.video_ids[index]
        grad_file = os.path.join(self.annotation_dir, f"{video_id}/{video_id}_gradient.npy")
        rand_file = os.path.join(self.annotation_dir, f"{video_id}/{video_id}_random.npy")
        bcknd_file = os.path.join(self.annotation_dir, f"{video_id}/{video_id}_background.npy")

        grad_annots = self._load_annotations(grad_file)
        rand_annots = self._load_annotations(rand_file)
        if os.path.exists(bcknd_file):
            bcknd_annots = self._load_annotations(bcknd_file)  # N1, T, 4
        else:
            _, T, _ = grad_annots.shape
            bcknd_annots = torch.empty((0, T, 4), dtype=torch.float32)

        grad_tracks = grad_annots[:, :, :2].unsqueeze(0)
        rand_tracks = rand_annots[:, :, :2].unsqueeze(0)
        bcknd_tracks = bcknd_annots[:, :, :2].unsqueeze(0)

        grad_vis = grad_annots[:, :, 2].unsqueeze(0)
        rand_vis = rand_annots[:, :, 2].unsqueeze(0)
        bcknd_vis = bcknd_annots[:, :, 2].unsqueeze(0)

        gt_tracks = torch.cat(
            [grad_tracks, rand_tracks, bcknd_tracks], dim=1
        )
        gt_vis = torch.cat([grad_vis, rand_vis, bcknd_vis], dim=1)
        queries = self._create_queries(gt_tracks)

        video_folder = os.path.join(self.video_dir, video_id)
        video_np = load_mose_video(video_folder)
        _, H, W, C = video_np.shape
        video_np = resize_video(video_np, (256, 256))
        
        video_tensor = torch.from_numpy(video_np).float()
        video_tensor = video_tensor.permute(0, 3, 1, 2)

        # Store in memory if needed
        gt_tracks[...,0] = gt_tracks[...,0] * (256 / W)
        gt_tracks[...,1] = gt_tracks[...,1] * (256 / H)

        queries[...,1] = queries[...,1] * (256 / W)
        queries[...,2] = queries[...,2] * (256 / H)
        
        return CoTrackerData(
            video_tensor,
            gt_tracks[0].permute(1, 0, 2),
            gt_vis[0].permute(1, 0),
            seq_name=str(video_id),
            query_points=queries[0],
        )