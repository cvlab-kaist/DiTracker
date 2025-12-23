# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from einops import rearrange

import torch
import torch.nn.functional as F

from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeBase, posenc
from cotracker.models.core.cotracker.feature_extractor import VDMEncoder

torch.manual_seed(0)


class DiTracker(CoTrackerThreeBase):
    def __init__(self, **args):
        super(DiTracker, self).__init__(**args)
        self.dit_fnet = VDMEncoder(
            model_path=self.model_path, 
            layer_hooks=self.layer_hooks, 
            head_hooks=self.head_hooks,
        )
        
        if not self.use_resnet:
            self.fnet = None

    def normalize_fmaps(self, fmaps):
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        return fmaps
    
    def get_pyramid_fmaps(self, fmaps):
        fmaps_pyramid = []
        fmaps_pyramid.append(fmaps)
        for i in range(self.corr_levels - 1):
            B, T, C, h, w = fmaps.shape
            fmaps_ = fmaps.reshape(
                B * T, C, h, w
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T, C, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)
        return fmaps_pyramid

    def forward(
        self,
        video,
        queries,
        iters=4,
        is_train=False,
        add_space_attn=True,
        fmaps_chunk_size=13,
    ):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - train_data: `None` if `is_train` is false, otherwise:
                - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                - mask (BoolTensor[B, T, N]):
        """
        B, T, C, H, W = video.shape
        device = queries.device
        assert H % self.stride == 0 and W % self.stride == 0

        B, N, __ = queries.shape
        # B = batch size
        # S_trimmed = actual number of frames in the window
        # N = number of tracks
        # C = color channels (3 for RGB)
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # video = B T C H W
        # queries = B N 3
        # coords_init = B T N 2
        # vis_init = B T N 1

        assert T >= 1  # A tracker needs at least two frames to track something
        video = video / 255.0
        dtype = video.dtype
        queried_frames = queries[:, :, 0].long()

        queried_coords = queries[..., 1:3].to(dtype=video.dtype)
        queried_coords = queried_coords / self.stride
        
        # We store our predictions here
        all_coords_predictions, all_vis_predictions, all_confidence_predictions = (
            [],
            [],
            [],
        )
        
        C_ = C
        if T > fmaps_chunk_size:
            # Chunk processing
            qfmaps, kfmaps = [], []
            for t in range(0, T, fmaps_chunk_size):
                video_chunk = torch.cat([video[:, :1], video[:, t : t + fmaps_chunk_size]], dim=1)
                qfmaps_chunk, kfmaps_chunk = self.dit_fnet(video_chunk)
                qfmaps.append(qfmaps_chunk[:, 1:])
                kfmaps.append(kfmaps_chunk[:, 1:])
                    
            qfmaps = torch.cat(qfmaps, dim=1)
            kfmaps = torch.cat(kfmaps, dim=1)
        else:
            qfmaps, kfmaps = self.dit_fnet(video)
            qfmaps, kfmaps = qfmaps[:, :T], kfmaps[:, :T]

        BK, T, c, h, w = qfmaps.shape
        
        fh = video.shape[-2] // self.stride
        fw = video.shape[-1] // self.stride
        qfmaps = F.interpolate(qfmaps.reshape(BK * T, c, h, w), size=(fh, fw), mode='bilinear', align_corners=False)
        kfmaps = F.interpolate(kfmaps.reshape(BK * T, c, h, w), size=(fh, fw), mode='bilinear', align_corners=False)
        
        qfmaps = qfmaps.reshape(BK, T, c, fh, fw)
        kfmaps = kfmaps.reshape(BK, T, c, fh, fw)
        
        layer_num = len(self.layer_hooks)

        # We compute track features
        qfmaps_pyramid = self.get_pyramid_fmaps(qfmaps) 
        kfmaps_pyramid = self.get_pyramid_fmaps(kfmaps)

        track_feat_pyramid = []
        track_feat_support_pyramid = []
        for i in range(self.corr_levels):
            track_feat, track_feat_support = self.get_track_feat(
                qfmaps_pyramid[i],
                queried_frames.repeat(layer_num, 1),
                queried_coords.repeat(layer_num, 1, 1) / 2**i,
                support_radius=self.corr_radius,
            )
            _, _, N, C = track_feat.shape
            track_feat_pyramid.append(track_feat.repeat(1, T, 1, 1))
            track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))

        if self.use_resnet:
            ### ResNet feature ###
            fmaps = self.fnet(video.reshape(-1, C_, H, W))
            fmaps = fmaps.reshape(
                B, T, -1, H // self.stride, W // self.stride
            )
            fmaps = fmaps.to(dtype)

            # We compute track features
            resnet_fmaps_pyramid = []
            resnet_track_feat_pyramid = []
            resnet_track_feat_support_pyramid = []
            resnet_fmaps_pyramid.append(fmaps)
            for i in range(self.corr_levels - 1):
                fmaps_ = fmaps.reshape(
                    B * T, -1, fmaps.shape[-2], fmaps.shape[-1]
                )
                fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
                fmaps = fmaps_.reshape(
                    B, T, -1, fmaps_.shape[-2], fmaps_.shape[-1]
                )
                resnet_fmaps_pyramid.append(fmaps)

            for i in range(self.corr_levels):
                track_feat, track_feat_support = self.get_track_feat(
                    resnet_fmaps_pyramid[i],
                    queried_frames,
                    queried_coords / 2**i,
                    support_radius=self.corr_radius,
                )
                resnet_track_feat_pyramid.append(track_feat.repeat(1, T, 1, 1))
                resnet_track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))


        D_coords = 2
        coord_preds, vis_preds, confidence_preds = [], [], []

        vis = torch.zeros((B, T, N), device=device).float()
        confidence = torch.zeros((B, T, N), device=device).float()
        coords = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2).float() #cost_coord_preds

        r = 2 * self.corr_radius + 1

        for it in range(iters):
            coords = coords.detach()  # B T N 2
            coords_init = coords.view(B * T, N, 2)
            corr_embs = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(
                    kfmaps_pyramid[i], coords_init.repeat(layer_num, 1, 1) / 2**i,
                )
                corr_feat = rearrange(corr_feat, "(b k) t n h w c -> b k t n h w c", b=B)
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B*layer_num, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                track_feat_support = rearrange(track_feat_support, "(b k) n i j c -> b k n i j c", b=B)
                corr_volume = torch.einsum(
                    "bktnhwc,bknijc->bktnhwij", corr_feat, track_feat_support
                ) / math.sqrt(self.latent_dim)
                corr_volume = F.softmax(corr_volume.flatten(-4, -3), dim=-3)
                corr_volume = corr_volume.mean(dim=1)
                corr_volume = corr_volume.view(B, T, N, r, r, r, r)
                
                if self.use_resnet:
                    corr_feat = self.get_correlation_feat(
                        resnet_fmaps_pyramid[i], coords_init / 2**i
                    )
                    track_feat_support = (
                        resnet_track_feat_support_pyramid[i]
                        .view(B, 1, r, r, N, -1)
                        .squeeze(1)
                        .permute(0, 3, 1, 2, 4)
                    ) # B N r r c
                    if not self.cost_softmax:
                        corr_feat = self.normalize_fmaps(corr_feat)
                        track_feat_support = self.normalize_fmaps(track_feat_support)

                    resnet_corr_volume = torch.einsum(
                        "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                    )
                    if self.cost_softmax:
                        resnet_corr_volume = resnet_corr_volume / math.sqrt(self.latent_dim)
                        resnet_corr_volume = F.softmax(resnet_corr_volume.flatten(-4, -3), dim=-3)
                        resnet_corr_volume = resnet_corr_volume.view(B, T, N, r, r, r, r)  
                
                if self.resnet_fuse_mode == "concat":
                    corr_volume_input = torch.cat([corr_volume.reshape(B*T*N,-1), resnet_corr_volume.reshape(B*T*N, -1)], dim=-1)
                elif self.resnet_fuse_mode == "add":
                    corr_volume_input = (corr_volume + resnet_corr_volume) / 2  
                    corr_volume_input = corr_volume_input.reshape(B*T*N, -1)
                else:
                    corr_volume_input = corr_volume.reshape(B*T*N, -1)
                
                corr_emb = self.corr_mlp(corr_volume_input)
                corr_embs.append(corr_emb)

                del corr_feat, track_feat_support, corr_emb, corr_volume
            
            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.view(B, T, N, corr_embs.shape[-1])

            transformer_input = [vis[..., None], confidence[..., None], corr_embs]

            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]

            rel_coords_forward = torch.nn.functional.pad(
                rel_coords_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_coords_backward = torch.nn.functional.pad(
                rel_coords_backward, (0, 0, 0, 0, 1, 0)
            )
            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]],
                    device=coords.device,
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )  # batch, num_points, num_frames, 84
            transformer_input.append(rel_pos_emb_input)

            x = (
                torch.cat(transformer_input, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, -1)
            )

            x = x + self.interpolate_time_embed(x, T)
            x = x.view(B, N, T, -1)  # (B N) T D -> B N T D

            del corr_embs
            delta = self.updateformer(
                x,
                add_space_attn=add_space_attn,
            )

            delta_coords = delta[..., :D_coords].permute(0, 2, 1, 3)
            delta_vis = delta[..., D_coords].permute(0, 2, 1)
            delta_confidence = delta[..., D_coords + 1].permute(0, 2, 1)

            vis = vis + delta_vis
            confidence = confidence + delta_confidence

            coords = coords + delta_coords
            coords_append = coords.clone()
            coords_append[..., :2] = coords_append[..., :2] * float(self.stride)
            
            coord_preds.append(coords_append)
            vis_preds.append(torch.sigmoid(vis))
            confidence_preds.append(torch.sigmoid(confidence))

            del transformer_input, rel_coords_forward, rel_coords_backward
            del rel_pos_emb_input, x, delta, delta_coords, delta_vis, delta_confidence
            del coords_append

        if is_train:
            all_coords_predictions.append([coord[..., :2] for coord in coord_preds])
            all_vis_predictions.append(vis_preds)
            all_confidence_predictions.append(confidence_preds)

        if is_train:
            train_data = (
                all_coords_predictions,
                all_vis_predictions,
                all_confidence_predictions,
                torch.ones_like(vis_preds[-1], device=vis_preds[-1].device),
            )
        else:
            train_data = None

        del kfmaps_pyramid, qfmaps_pyramid, track_feat_support_pyramid
        del qfmaps, kfmaps
        del vis, confidence, coords

        return coord_preds[-1][..., :2], vis_preds[-1], confidence_preds[-1], train_data