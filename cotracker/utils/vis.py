# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import imageio
import torch

from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision
import matplotlib as mpl

import torch
import torchvision
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import os

from PIL import Image
def concat_variant_across_i(src_path, tgt_path, variant_name, i_values, output_name):
    imgs = []
    for i in i_values:
        img = Image.open(f"{src_path}/{variant_name}{i}.png").convert("RGB")
        imgs.append(np.array(img))

    # horizontally concatenate
    concat = np.concatenate(imgs, axis=1)
    concat_img = Image.fromarray(concat)
    os.makedirs(tgt_path, exist_ok=True)
    concat_img.save(f"{tgt_path}/{output_name}.png")
    

def _safe_overlay_patch(
    img: torch.Tensor,         # (B,C,H,W)
    patch: torch.Tensor,       # (C, win, win) on same device
    center_xy: torch.Tensor,   # (B,2) longs [x,y] in pixels
    half_win: int,
    alpha: float = 0.7,
):
    """
    Overlays 'patch' centered at center_xy[b] for each batch element.
    If the window falls (partly) outside the image, it crops both
    image ROI and patch ROI accordingly. If the intersection is empty,
    it skips that element.
    """
    B, C, H, W = img.shape
    HP, WP = patch.shape[-2], patch.shape[-1]
    assert HP == 2 * half_win + 1 and WP == 2 * half_win + 1, \
        f"patch size {(HP,WP)} must match win={2*half_win+1}"

    for b in range(B):
        x = int(center_xy[b, 0].item())
        y = int(center_xy[b, 1].item())

        # If center is far outside the image, skip early
        if x < -half_win or x > W - 1 + half_win or y < -half_win or y > H - 1 + half_win:
            continue

        # Image ROI (exclusive hi)
        x0 = max(0, x - half_win)
        x1 = min(W, x + half_win + 1)
        y0 = max(0, y - half_win)
        y1 = min(H, y + half_win + 1)

        if x0 >= x1 or y0 >= y1:
            # Empty intersection -> skip
            continue

        # Matching patch ROI
        # How many pixels did we clip on each side?
        px0 = half_win - (x - x0)
        py0 = half_win - (y - y0)
        px1 = px0 + (x1 - x0)
        py1 = py0 + (y1 - y0)

        patch_crop = patch[:, py0:py1, px0:px1]  # (C, y1-y0, x1-x0)
        img[b:b+1, :, y0:y1, x0:x1] = (
            img[b:b+1, :, y0:y1, x0:x1] * (1.0 - alpha)
            + patch_crop.unsqueeze(0) * alpha
        )
    return img

def visualize_corr_volume(
    corr_volume, 
    video, 
    coords, 
    queried_coords, 
    queried_frames, 
    stride, i, save_path, target_point = 0, key_frame = 20
):
    corr_volume = corr_volume.to(torch.float32)  # (B, T, N, r, r, r, r) expected
    B, T, C, H, W = video.shape

    # pick query and key frames
    target_frame = queried_frames[:, target_point] # (B,)
    query_image = video[torch.arange(B), target_frame] # (B,C,H,W)
    key_image = video[torch.arange(B), key_frame] # (B,C,H,W)

    # normalize per-batch image for visualization
    def _minmax(x):
        xmin = x.amin(dim=(1,2,3), keepdim=True)
        xmax = x.amax(dim=(1,2,3), keepdim=True)
        return (x - xmin) / (xmax - xmin + 1e-8)
    query_image = _minmax(query_image)
    key_image = _minmax(key_image)

    # dynamic center index from corr volume
    # corr_volume[b, key_frame, target_point] : (r, r, r, r)
    # we take central slice along the last 2 dims to visualize spatial key-plane correlation
    _, _, _, r1, r2, r3, r4 = corr_volume.shape
    cy = r3 // 2
    cx = r4 // 2
    corr_volume_vis = corr_volume[torch.arange(B), key_frame, target_point][:, :, :, cy, cx]  # (B, r1, r2)

    # compute window size to overlay (must be odd)
    half_win = 3 * 4 * (2 ** i)  # your original formula (window half size)
    win = 2 * half_win + 1

    # resize heatmap to match window size
    # corr_volume_vis: (B, r1, r2) -> normalize -> color map -> get (B,3,win,win)
    corr_np = corr_volume_vis.detach().cpu().numpy()  # (B, r, r)

    # normalize heatmap per-batch for min-max map
    colormapped_list = []
    cos_colormapped_list = []
    for b in range(B):
        a = corr_np[b]
        # min-max
        normalizer = mpl.colors.Normalize(vmin=a.min(), vmax=a.max())
        mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
        rgb = (mapper.to_rgba(a)[:, :, :3] * 255).astype(np.uint8)       # (r, r, 3)

        # cosine-like symmetric mapping (if your values are in [-1,1], else this is just a style)
        cos_norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
        cos_mapper = cm.ScalarMappable(norm=cos_norm, cmap='viridis')
        rgb_cos = (cos_mapper.to_rgba(a)[:, :, :3] * 255).astype(np.uint8)

        # resize to (win, win)
        t_rgb     = torchvision.transforms.functional.resize(
            torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float(), size=[win, win],
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )[0] / 255.0  # (3,win,win)
        t_rgb_cos = torchvision.transforms.functional.resize(
            torch.from_numpy(rgb_cos).permute(2,0,1).unsqueeze(0).float(), size=[win, win],
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )[0] / 255.0

        colormapped_list.append(t_rgb)
        cos_colormapped_list.append(t_rgb_cos)

    colormapped = torch.stack(colormapped_list, dim=0).to(key_image.device)      # (B,3,win,win)
    cos_mapped  = torch.stack(cos_colormapped_list, dim=0).to(key_image.device)  # (B,3,win,win)

    # compute integer pixel points
    key_point   = (coords[:, key_frame, target_point] * stride).round().long()  # (B,2)
    query_point = (queried_coords[:, target_point] * stride).round().long()  # (B,2)

    # overlay (handles borders)
    os.makedirs(save_path, exist_ok=True)
    key_img_minmax = key_image.clone()
    key_img_cos    = key_image.clone()
    
    # Overlay (per batch)
    for variant_name, patch_stack, out_img in [
        ("minmax_corr", colormapped, key_img_minmax),
        ("cosine_corr", cos_mapped,  key_img_cos),
    ]:
        for b in range(B):
            out_img[b:b+1] = _safe_overlay_patch(
                out_img[b:b+1],                # (1,C,H,W) slice is fine
                patch_stack[b],                # (3,win,win)
                key_point[b:b+1],              # (1,2)
                half_win,
                alpha=0.7,
            )
        torchvision.utils.save_image(out_img, f"{save_path}/{variant_name}{i}.png")

    # draw a small white square on the query image at query_point (clamped)
    qimg = query_image.clone()
    sq = 5
    for b in range(B):
        x = int(query_point[b, 0].item())
        y = int(query_point[b, 1].item())
        y0, y1 = max(0, y - sq), min(H - 1, y + sq) + 1
        x0, x1 = max(0, x - sq), min(W - 1, x + sq) + 1
        qimg[b, :, y0:y1, x0:x1] = 1.0
    torchvision.utils.save_image(qimg, f"{save_path}/query_frame.png")


def read_video_from_path(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)


def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb


def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb


def add_weighted(rgb, alpha, original, beta, gamma):
    return (rgb * alpha + original * beta + gamma).astype("uint8")


class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 2,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame=0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        opacity: float = 1.0,
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        color_alpha = int(opacity * 255)
        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            color_alpha=color_alpha,
        )
        if save_video:
            self.save_video(res_video, filename=filename, writer=writer, step=step)
        return res_video

    def save_video(self, video, filename, writer=None, step=0):
        if writer is not None:
            writer.add_video(
                filename,
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
            
            # Also save each rendered frame as an image inside a frames/ subfolder
            frames_dir = os.path.join(self.save_dir, filename, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for idx, frame in enumerate(wide_list):
                Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
            
            # Prepare the video file path
            save_path = os.path.join(self.save_dir, filename, f"video.mp4")

            # Create a writer object
            video_writer = imageio.get_writer(save_path, fps=self.fps)

            # Write frames to the video file
            for frame in wide_list[2:-1]:
                video_writer.append_data(frame)

            video_writer.close()

            print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame=0,
        compensate_for_camera_motion=False,
        color_alpha: int = 255,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()
        if visibility is not None:
            visibility = visibility[0].detach().cpu().numpy()  # T, N, 1 or T, N
            if visibility.ndim == 3:
                visibility = visibility.squeeze(-1)  # T, N

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())
        vector_colors = np.zeros((T, N, 3))

        if self.mode == "optical_flow":
            import flow_vis

            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])
        elif segm_mask is None:
            if self.mode == "rainbow":
                # Use track index for consistent coloring across GT and pred
                # This ensures the same track index always gets the same color
                n_min, n_max = 0, N - 1
                if N > 1:
                    norm = plt.Normalize(n_min, n_max)
                    for n in range(N):
                        # Color based on track index (n) instead of Y-coordinate
                        color = self.color_map(norm(n))
                        color = np.array(color[:3])[None] * 255
                        vector_colors[:, n] = np.repeat(color, T, axis=0)
                else:
                    # Single track case
                    color = np.array(self.color_map(0.5)[:3])[None] * 255
                    vector_colors[:, 0] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                # Use track index for consistent coloring across GT and pred
                n_min, n_max = 0, N - 1
                if N > 1:
                    norm = plt.Normalize(n_min, n_max)
                    for n in range(N):
                        if segm_mask[n] > 0:
                            # Color based on track index (n) instead of Y-coordinate
                            color = self.color_map(norm(n))
                            color = np.array(color[:3])[None] * 255
                            vector_colors[:, n] = np.repeat(color, T, axis=0)
                else:
                    # Single track case
                    if segm_mask[0] > 0:
                        color = np.array(self.color_map(0.5)[:3])[None] * 255
                        vector_colors[:, 0] = np.repeat(color, T, axis=0)

            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(query_frame + 1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                # Set coordinates to negative when occluded to prevent drawing trajectory
                if visibility is not None:
                    curr_visibility = visibility[first_ind : t + 1]  # T_curr, N
                    if compensate_for_camera_motion:
                        curr_visibility = curr_visibility[:, segm_mask > 0]
                    # Set coordinates to negative when not visible (occluded)
                    occluded_mask = ~curr_visibility.astype(bool)
                    curr_tracks[:, :, 0][occluded_mask] = -1
                    curr_tracks[:, :, 1][occluded_mask] = -1

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind : t + 1]
                    )

        #  draw points
        for t in range(T):
            img = Image.fromarray(np.uint8(res_video[t]))
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[t, i]
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (
                        compensate_for_camera_motion and segm_mask[i] > 0
                    ):
                        img = draw_circle(
                            img,
                            coord=coord,
                            radius=int(self.linewidth),
                            color=vector_colors[t, i].astype(int),
                            visible=visibile,
                            color_alpha=color_alpha,
                        )
            res_video[t] = np.array(img)

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.3,
    ):
        T, N, _ = tracks.shape
        rgb = Image.fromarray(np.uint8(rgb))
        active_traces = (tracks[0, :, 0] > 0) & (tracks[0, :, 1] > 0)
        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 1.5
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    if (
                        coord_y[0] <= 0
                        or coord_y[1] <= 0
                        or coord_x[0] <= 0
                        or coord_x[1] <= 0
                    ):
                        active_traces[i] = False
                        continue
                    if not active_traces[i]:
                        active_traces[i] = True
                        continue
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].astype(int),
                        self.linewidth + 1,
                    )
                else:
                    active_traces[i] = False
            if self.tracks_leave_trace > 0:
                rgb = Image.fromarray(
                    np.uint8(
                        add_weighted(
                            np.array(rgb), alpha, np.array(original), 1 - alpha, 0
                        )
                    )
                )
        rgb = np.array(rgb)
        return rgb

    def _draw_gt_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3,
        gt_tracks: np.ndarray,  # T x 2
    ):
        T, N, _ = gt_tracks.shape
        color = np.array((211, 0, 0))
        rgb = Image.fromarray(np.uint8(rgb))
        for t in range(T):
            for i in range(N):
                gt_tracks = gt_tracks[t][i]
                #  draw a red cross
                if gt_tracks[0] > 0 and gt_tracks[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (int(gt_tracks[0]) + length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) - length, int(gt_tracks[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth + 1,
                    )
                    coord_y = (int(gt_tracks[0]) - length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) + length, int(gt_tracks[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
        rgb = np.array(rgb)
        return rgb
