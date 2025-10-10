import os
import sys
sys.path.append(os.getcwd())
import yaml
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
from models.autoencoder_kl import AutoencoderKL

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalVAEMixin
from diffusers.utils.torch_utils import is_torch_version
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder

from models.RTS_blocks import RTSModule

class CustomVidEncoder(Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # down
        # self.video_down_RTS_blocks = nn.ModuleList([])
        # for i in range(len(self.down_blocks)):
        #     output_channel = block_out_channels[i]
        #     self.video_down_RTS_blocks.append(RTSModule(nhidden=output_channel))
        self.inference_num_frames_per_batch = None
        self.temporal_device = None

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""

        if self.training and self.gradient_checkpointing:
            sample = self.conv_in(sample)

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                # for down_block, RTS_block in zip(self.down_blocks, self.video_down_RTS_blocks):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                    # sample = torch.utils.checkpoint.checkpoint(
                    #     create_custom_forward(RTS_block), sample, use_reentrant=False
                    # )
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                # for down_block, RTS_block in zip(self.down_blocks, self.video_down_RTS_blocks):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                    # sample = torch.utils.checkpoint.checkpoint(create_custom_forward(RTS_block), sample)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

            # post-process
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)

        elif self.inference_num_frames_per_batch is not None:
            # down + inference
            assert self.inference_num_frames_per_batch > 0, "inference_num_frames_per_batch should be greater than 0"
            bs_inference = math.ceil(sample.shape[0]/(self.inference_num_frames_per_batch))
            sample_memory = []
            t_, c_, h_, w_ = sample.shape
            for i in range(bs_inference):
                sample_ = sample[i*self.inference_num_frames_per_batch:min((i+1)*self.inference_num_frames_per_batch,t_), :, :, :]
                sample_ = self.conv_in(sample_)
                sample_memory.append(sample_)
            
            # for down_block, RTS_block in zip(self.down_blocks, self.video_down_RTS_blocks):
            for down_block in self.down_blocks:
                for i in range(bs_inference):
                    sample_memory[i] = down_block(sample_memory[i])
                # sample_memory = RTS_block(sample_memory, inference_num_frames_per_batch=self.inference_num_frames_per_batch, device=self.temporal_device)

            # middle
            for i in range(bs_inference):
                sample_memory[i] = self.mid_block(sample_memory[i])
                # post-process
                sample_memory[i] = self.conv_norm_out(sample_memory[i])
                sample_memory[i] = self.conv_act(sample_memory[i])
                sample_memory[i] = self.conv_out(sample_memory[i])
            sample = torch.cat(sample_memory, dim=0)
        else:
            sample = self.conv_in(sample)
            # down
            # for down_block, RTS_block in zip(self.down_blocks, self.video_down_RTS_blocks):
            for down_block in self.down_blocks:
                sample = down_block(sample)
                # sample = RTS_block(sample)
            # middle
            sample = self.mid_block(sample)

            # post-process
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)

        return sample

class CustomVidDecoder(Decoder):
    def __init__(self, block_out_channels, *args, **kwargs):
        super().__init__(block_out_channels=block_out_channels, *args, **kwargs)
        # up
        self.video_up_RTS_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(self.up_blocks)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            self.video_up_RTS_blocks.append(RTSModule(nhidden=prev_output_channel))
        self.inference_num_frames_per_batch = None
        self.temporal_device = None

    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:
            sample = self.conv_in(sample)

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block, RTS_block in zip(self.up_blocks, self.video_up_RTS_blocks):
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(RTS_block),
                        sample,
                        use_reentrant=False,
                    )
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
                    
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block, RTS_block in zip(self.up_blocks, self.video_up_RTS_blocks):
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(RTS_block), sample)
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
            # post-process
            if latent_embeds is None:
                sample = self.conv_norm_out(sample)
            else:
                sample = self.conv_norm_out(sample, latent_embeds)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)

        elif self.inference_num_frames_per_batch is not None:
            assert self.inference_num_frames_per_batch > 0, "inference_num_frames_per_batch should be greater than 0"
            bs_inference = math.ceil(sample.shape[0]/(self.inference_num_frames_per_batch))
            sample_memory = []
            t_, c_, h_, w_ = sample.shape
            for i in range(bs_inference):
                sample_ = sample[i*self.inference_num_frames_per_batch:min((i+1)*self.inference_num_frames_per_batch,t_), :, :, :]
                sample_ = self.conv_in(sample_)
                sample_memory.append(sample_)

            # middle
            for i in range(bs_inference):
                sample_memory[i] = self.mid_block(sample_memory[i], latent_embeds)
                sample_memory[i] = sample_memory[i].to(upscale_dtype)

            # up
            for up_block, RTS_block in zip(self.up_blocks, self.video_up_RTS_blocks):
                sample_memory = RTS_block(sample_memory, inference_num_frames_per_batch=self.inference_num_frames_per_batch, device=self.temporal_device)
                for i in range(bs_inference):
                    sample_memory[i] = up_block(sample_memory[i], latent_embeds)

            # post-process
            for i in range(bs_inference):
                if latent_embeds is None:
                    sample_memory[i] = self.conv_norm_out(sample_memory[i])
                else:
                    sample_memory[i] = self.conv_norm_out(sample_memory[i], latent_embeds)
                sample_memory[i] = self.conv_act(sample_memory[i])
                sample_memory[i] = self.conv_out(sample_memory[i])
            sample = torch.cat(sample_memory, dim=0)
        else:
            sample = self.conv_in(sample)
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for up_block, RTS_block in zip(self.up_blocks, self.video_up_RTS_blocks):
                sample = RTS_block(sample)
                sample = up_block(sample, latent_embeds)

            # post-process
            if latent_embeds is None:
                sample = self.conv_norm_out(sample)
            else:
                sample = self.conv_norm_out(sample, latent_embeds)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)

        return sample

class CustomVidAutoencoderKL(AutoencoderKL):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        *args, **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            *args, **kwargs
            )

        # pass init params to Encoder
        self.encoder = CustomVidEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = CustomVidDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )
