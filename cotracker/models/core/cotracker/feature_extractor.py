import torch
from torch import nn

from einops import rearrange
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
    CogVideoXDPMScheduler
)

class VDMEncoder(nn.Module):
    def __init__(
        self, 
        model_path: str = "THUDM/CogVideoX-2B",
        layer_hooks: list[int] = [17],
        head_hooks: list[int] = [2],
    ):
        super(VDMEncoder, self).__init__()
        
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer"
        )
        self.feature_dim = self.transformer.config.attention_head_dim
        self.layer_hooks = layer_hooks
        self.head_hooks = head_hooks

        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_path,
            subfolder="vae"
        )
        self.vae.enable_slicing()
        self.vae.enable_tiling()

        self.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        prompt_token_ids = tokenizer(
            "",
            padding="max_length",
            max_length=self.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        self.prompt_embedding = text_encoder(prompt_token_ids)[0].detach()

    # Encode each frame of video independently
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, T, H, W]
        latent_dist = [self.vae.encode(video[:,:,i].unsqueeze(2)).latent_dist for i in range(video.size(2))]
        latent = [dist.sample() * self.vae.config.scaling_factor for dist in latent_dist]
        latent = torch.cat(latent, dim=2)
        latent = latent * self.scheduler.init_noise_sigma
        
        return latent

    def forward(self, x):
        latent = self.encode_video(x.permute(0, 2, 1, 3, 4)) # [B, T, C, H, W]
        latent = latent.to(dtype=self.transformer.dtype, device=self.transformer.device) 
        B, T, C, img_h, img_w = x.shape

        patch_size_t = self.transformer.config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        # Get prompt embeddings
        prompt_embedding = self.prompt_embedding.expand(B, -1, -1).to(dtype=latent.dtype, device=latent.device)

        # Add noise to latent
        latent = latent.permute(0, 2, 1, 3, 4)  # from [B, C, T, H, W] to [B, T, C, H, W]
        latent = self.scheduler.scale_model_input(latent, 0)

        timesteps = torch.tensor([19] * B).to(dtype=latent.dtype, device=latent.device)
        timesteps = timesteps.long()

        _, queries_, keys_ = self.transformer(
            hidden_states=latent,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=None,
            return_dict=False,
            layer_hooks=self.layer_hooks
        )
        # batch_size, head_num, fhw, feat_dim = queries[0].shape
        f_h = img_h // (self.transformer.config.patch_size * 8)
        f_w = img_w // (self.transformer.config.patch_size * 8)

        queries = [
            rearrange(queries_[layer][:, self.head_hooks[idx]], "b (t fh fw) c -> b t c fh fw", fh=f_h, fw=f_w)
            for idx, layer in enumerate(self.layer_hooks)
        ]
        queries = torch.cat(queries, dim=0)
        keys = [
            rearrange(keys_[layer][:, self.head_hooks[idx]], "b (t fh fw) c -> b t c fh fw", fh=f_h, fw=f_w)
            for idx, layer in enumerate(self.layer_hooks)
        ]
        keys = torch.cat(keys, dim=0)
        
        return queries, keys