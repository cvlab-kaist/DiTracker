# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Sequence

EPS = 1e-6


def smart_cat(tensor1, tensor2, dim):
    if tensor1 is None:
        return tensor2
    return torch.cat([tensor1, tensor2], dim=dim)


def get_uniformly_sampled_pts(
    size: int,
    num_frames: int,
    extent: Tuple[float, ...],
    device: Optional[torch.device] = torch.device("cpu"),
):
    time_points = torch.randint(low=0, high=num_frames, size=(size, 1), device=device)
    space_points = torch.rand(size, 2, device=device) * torch.tensor(
        [extent[1], extent[0]], device=device
    )
    points = torch.cat((time_points, space_points), dim=1)
    return points[None]


def get_superpoint_sampled_pts(
    video,
    size: int,
    num_frames: int,
    extent: Tuple[float, ...],
    device: Optional[torch.device] = torch.device("cpu"),
):
    extractor = SuperPoint(max_num_keypoints=48).eval().cuda()
    points = list()
    for _ in range(8):
        frame_num = random.randint(0, int(num_frames * 0.25))
        key_points = extractor.extract(
            video[0, frame_num, :, :, :] / 255.0, resize=None
        )["keypoints"]
        frame_tensor = torch.full((1, key_points.shape[1], 1), frame_num).cuda()
        points.append(torch.cat([frame_tensor.cuda(), key_points], dim=2))
    return torch.cat(points, dim=1)[:, :size, :]


def get_sift_sampled_pts(
    video,
    size: int,
    num_frames: int,
    extent: Tuple[float, ...],
    device: Optional[torch.device] = torch.device("cpu"),
    num_sampled_frames: int = 8,
    sampling_length_percent: float = 0.25,
):
    import cv2
    # assert size == 384, "hardcoded for experiment"
    sift = cv2.SIFT_create(nfeatures=size // num_sampled_frames)
    points = list()
    for _ in range(num_sampled_frames):
        frame_num = random.randint(0, int(num_frames * sampling_length_percent))
        key_points, _ = sift.detectAndCompute(
            video[0, frame_num, :, :, :]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
            .astype(np.uint8),
            None,
        )
        for kp in key_points:
            points.append([frame_num, int(kp.pt[0]), int(kp.pt[1])])
    return torch.tensor(points[:size], device=device)[None]


def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    r"""Masked mean

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """

    mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (EPS + denom)
    return mean


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
        )
    else:
        coords = coords * torch.tensor(
            [2 / size for size in reversed(sizes)], device=coords.device
        )

    coords -= 1

    return F.grid_sample(
        input, coords, align_corners=align_corners, padding_mode=padding_mode
    )


def sample_features4d(input, coords):
    r"""Sample spatial features

    `sample_features4d(input, coords)` samples the spatial features
    :attr:`input` represented by a 4D tensor :math:`(B, C, H, W)`.

    The field is sampled at coordinates :attr:`coords` using bilinear
    interpolation. :attr:`coords` is assumed to be of shape :math:`(B, R,
    3)`, where each sample has the format :math:`(x_i, y_i)`. This uses the
    same convention as :func:`bilinear_sampler` with `align_corners=True`.

    The output tensor has one feature per point, and has shape :math:`(B,
    R, C)`.

    Args:
        input (Tensor): spatial features.
        coords (Tensor): points.

    Returns:
        Tensor: sampled features.
    """

    B, _, _, _ = input.shape

    # B R 2 -> B R 1 2
    coords = coords.unsqueeze(2)

    # B C R 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 1, 3).view(
        B, -1, feats.shape[1] * feats.shape[3]
    )  # B C R 1 -> B R C


def sample_features5d(input, coords):
    r"""Sample spatio-temporal features

    `sample_features5d(input, coords)` works in the same way as
    :func:`sample_features4d` but for spatio-temporal features and points:
    :attr:`input` is a 5D tensor :math:`(B, T, C, H, W)`, :attr:`coords` is
    a :math:`(B, R1, R2, 3)` tensor of spatio-temporal point :math:`(t_i,
    x_i, y_i)`. The output tensor has shape :math:`(B, R1, R2, C)`.

    Args:
        input (Tensor): spatio-temporal features.
        coords (Tensor): spatio-temporal points.

    Returns:
        Tensor: sampled features.
    """

    B, T, _, _, _ = input.shape

    # B T C H W -> B C T H W
    input = input.permute(0, 2, 1, 3, 4)

    # B R1 R2 3 -> B R1 R2 1 3
    coords = coords.unsqueeze(3)

    # B C R1 R2 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 3, 1, 4).view(
        B, feats.shape[2], feats.shape[3], feats.shape[1]
    )  # B C R1 R2 1 -> B R1 R2 C


def get_grid(
    height,
    width,
    shape=None,
    dtype="torch",
    device="cpu",
    align_corners=True,
    normalize=True,
):
    H, W = height, width
    S = shape if shape else []
    if align_corners:
        x = torch.linspace(0, 1, W, device=device)
        y = torch.linspace(0, 1, H, device=device)
        if not normalize:
            x = x * (W - 1)
            y = y * (H - 1)
    else:
        x = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device)
        y = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device)
        if not normalize:
            x = x * W
            y = y * H
    x_view, y_view, exp = [1 for _ in S] + [1, -1], [1 for _ in S] + [-1, 1], S + [H, W]
    x = x.view(*x_view).expand(*exp)
    y = y.view(*y_view).expand(*exp)
    grid = torch.stack([x, y], dim=-1)
    if dtype == "numpy":
        grid = grid.numpy()
    return grid


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
        )
    else:
        coords = coords * torch.tensor(
            [2 / size for size in reversed(sizes)], device=coords.device
        )

    coords -= 1

    return F.grid_sample(
        input, coords, align_corners=align_corners, padding_mode=padding_mode
    )


def round_to_multiple_of_4(n):
    return round(n / 4) * 4


# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

def bilinear(x: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
  """Resizes a 5D tensor using bilinear interpolation.

  Args:
        x: A 5D tensor of shape (B, T, W, H, C) where B is batch size, T is
          time, W is width, H is height, and C is the number of channels.
    resolution: The target resolution as a tuple (new_width, new_height).

  Returns:
    The resized tensor.
  """
  b, t, h, w, c = x.size()
  x = x.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)
  x = F.interpolate(x, size=resolution, mode='bilinear', align_corners=False)
  b, _, h, w = x.size()
  x = x.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)
  return x


def map_coordinates_3d(
    feats: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
  """Maps 3D coordinates to corresponding features using bilinear interpolation.

  Args:
    feats: A 5D tensor of features with shape (B, W, H, D, C), where B is batch
      size, W is width, H is height, D is depth, and C is the number of
      channels.
    coordinates: A 3D tensor of coordinates with shape (B, N, 3), where N is the
      number of coordinates and the last dimension represents (W, H, D)
      coordinates.

  Returns:
    The mapped features tensor.
  """
  x = feats.permute(0, 4, 1, 2, 3)
  y = coordinates[:, :, None, None, :].float().clone()
  y[..., 0] = y[..., 0] + 0.5
  y = 2 * (y / torch.tensor(x.shape[2:], device=y.device)) - 1
  y = torch.flip(y, dims=(-1,))
  out = (
      F.grid_sample(
          x, y, mode='bilinear', align_corners=False, padding_mode='border'
      )
      .squeeze(dim=(3, 4))
      .permute(0, 2, 1)
  )
  return out


def map_coordinates_2d(
    feats: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
  """Maps 2D coordinates to feature maps using bilinear interpolation.

  The function performs bilinear interpolation on the feature maps (`feats`)
  at the specified `coordinates`. The coordinates are normalized between
  -1 and 1 The result is a tensor of sampled features corresponding
  to these coordinates.

  Args:
    feats (Tensor): A 5D tensor of shape (N, T, H, W, C) representing feature
      maps, where N is the batch size, T is the number of frames, H and W are
      height and width, and C is the number of channels.
    coordinates (Tensor): A 5D tensor of shape (N, P, T, S, XY) representing
      coordinates, where N is the batch size, P is the number of points, T is
      the number of frames, S is the number of samples, and XY represents the 2D
      coordinates.

  Returns:
    Tensor: A 5D tensor of the sampled features corresponding to the
      given coordinates, of shape (N, P, T, S, C).
  """
  n, t, h, w, c = feats.shape
  x = feats.permute(0, 1, 4, 2, 3).view(n * t, c, h, w)

  n, p, t, s, xy = coordinates.shape
  y = coordinates.permute(0, 2, 1, 3, 4).reshape(n * t, p, s, xy)
  y = 2 * (y / torch.tensor([h, w], device=feats.device)) - 1
  y = torch.flip(y, dims=(-1,)).float()

  out = F.grid_sample(
      x, y, mode='bilinear', align_corners=False, padding_mode='zeros'
  )
  _, c, _, _ = out.shape
  out = out.permute(0, 2, 3, 1).view(n, t, p, s, c).permute(0, 2, 1, 3, 4)

  return out


def soft_argmax_heatmap_batched(softmax_val, threshold=5):
  """Test if two image resolutions are the same."""
  b, n, t, h, w = softmax_val.shape
  y, x = torch.meshgrid(
      torch.arange(h, device=softmax_val.device),
      torch.arange(w, device=softmax_val.device),
      indexing='ij',
  )
  coords = torch.stack([x + 0.5, y + 0.5], dim=-1).to(softmax_val.device)
  softmax_val_flat = softmax_val.reshape(b, n, t, -1)
  argmax_pos = torch.argmax(softmax_val_flat, dim=-1)

  pos = coords.reshape(-1, 2)[argmax_pos]
  valid = (
      torch.sum(
          torch.square(
              coords[None, None, None, :, :, :] - pos[:, :, :, None, None, :]
          ),
          dim=-1,
          keepdims=True,
      )
      < threshold**2
  )

  weighted_sum = torch.sum(
      coords[None, None, None, :, :, :]
      * valid
      * softmax_val[:, :, :, :, :, None],
      dim=(3, 4),
  )
  sum_of_weights = torch.maximum(
      torch.sum(valid * softmax_val[:, :, :, :, :, None], dim=(3, 4)),
      torch.tensor(1e-12, device=softmax_val.device),
  )
  return weighted_sum / sum_of_weights


def heatmaps_to_points(
    all_pairs_softmax,
    image_shape,
    threshold=5,
    query_points=None,
):
  """Convert heatmaps to points using soft argmax."""

  out_points = soft_argmax_heatmap_batched(all_pairs_softmax, threshold)
  feature_grid_shape = all_pairs_softmax.shape[1:]
  # Note: out_points is now [x, y]; we need to divide by [width, height].
  # image_shape[3] is width and image_shape[2] is height.
  out_points = convert_grid_coordinates(
      out_points,
      feature_grid_shape[3:1:-1],
      image_shape[3:1:-1],
  )
  assert feature_grid_shape[1] == image_shape[1]
  if query_points is not None:
    # The [..., 0:1] is because we only care about the frame index.
    query_frame = convert_grid_coordinates(
        query_points.detach(),
        image_shape[1:4],
        feature_grid_shape[1:4],
        coordinate_format='tyx',
    )[..., 0:1]

    query_frame = torch.round(query_frame)
    frame_indices = torch.arange(image_shape[1], device=query_frame.device)[
        None, None, :
    ]
    is_query_point = query_frame == frame_indices

    is_query_point = is_query_point[:, :, :, None]
    out_points = (
        out_points * ~is_query_point
        + torch.flip(query_points[:, :, None], dims=(-1,))[..., 0:2]
        * is_query_point
    )

  return out_points


def is_same_res(r1, r2):
  """Test if two image resolutions are the same."""
  return all([x == y for x, y in zip(r1, r2)])


def convert_grid_coordinates(
    coords: torch.Tensor,
    input_grid_size: Sequence[int],
    output_grid_size: Sequence[int],
    coordinate_format: str = 'xy',
) -> torch.Tensor:
  """Convert grid coordinates to correct format."""
  if isinstance(input_grid_size, tuple):
    input_grid_size = torch.tensor(input_grid_size, device=coords.device)
  if isinstance(output_grid_size, tuple):
    output_grid_size = torch.tensor(output_grid_size, device=coords.device)

  if coordinate_format == 'xy':
    if input_grid_size.shape[0] != 2 or output_grid_size.shape[0] != 2:
      raise ValueError(
          'If coordinate_format is xy, the shapes must be length 2.'
      )
  elif coordinate_format == 'tyx':
    if input_grid_size.shape[0] != 3 or output_grid_size.shape[0] != 3:
      raise ValueError(
          'If coordinate_format is tyx, the shapes must be length 3.'
      )
    if input_grid_size[0] != output_grid_size[0]:
      raise ValueError('converting frame count is not supported.')
  else:
    raise ValueError('Recognized coordinate formats are xy and tyx.')

  position_in_grid = coords
  position_in_grid = position_in_grid * output_grid_size / input_grid_size

  return position_in_grid
