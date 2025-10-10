import os, sys
sys.path.append(os.getcwd())
import math
import pyiqa
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from models.autoencoder_kl_with_RTS import CustomVidAutoencoderKL ## temporal-shift
from models.unet_2d_with_RTS import UNet2DConditionVideoModel ## temporal-shift
from peft import LoraConfig
import matplotlib.pyplot as plt

class UltraVSR_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        if torch.cuda.is_available():   ## temporal-shift
            self.device = torch.device('cuda:0')
            if torch.cuda.device_count() >= 2:
                self.temporal_device = torch.device('cuda:1')
            else:
                self.temporal_device = None ## same as self.device

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_diffusion_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_diffusion_model_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_diffusion_model_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.vae = CustomVidAutoencoderKL.from_pretrained(self.args.pretrained_diffusion_model_path, subfolder="vae", low_cpu_mem_usage=False)
        self.unet = UNet2DConditionVideoModel.from_pretrained(self.args.pretrained_diffusion_model_path, subfolder="unet", low_cpu_mem_usage=False)   ## temporal-shift

        self.vae.encoder.inference_num_frames_per_batch = self.args.vae_encoder_num_frames_per_batch
        self.vae.decoder.inference_num_frames_per_batch = self.args.vae_decoder_num_frames_per_batch
        self.unet.inference_num_frames_per_batch=self.args.latent_num_frames_per_batch
        self.metric_model = pyiqa.create_metric('clipiqa')

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        UltraVSR_weight = torch.load(args.UltraVSR_weight_path)
        self.load_ckpt(UltraVSR_weight)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        

         ## temporal-shift
        if self.temporal_device is not None:
            for block in self.unet.video_down_RTS_blocks:
                block.to(self.temporal_device)
            # self.unet.video_mid_RTS_block.to(self.temporal_device)
            for block in self.unet.video_up_RTS_blocks:
                block.to(self.temporal_device)
            self.unet.temporal_device = self.temporal_device

            # for block in self.vae.encoder.video_down_RTS_blocks:
            #     block.to(self.temporal_device)
            for block in self.vae.decoder.video_up_RTS_blocks:
                block.to(self.temporal_device)
            self.vae.encoder.temporal_device = self.temporal_device
            self.vae.decoder.temporal_device = self.temporal_device

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    # @perfcount
    @torch.no_grad()
    def forward(self, lq, degra_factor, prompts):
        degra_timesteps = torch.argmin(torch.abs(degra_factor - self.noise_scheduler.alphas_cumprod)).item()
        degra_timesteps = torch.tensor([degra_timesteps], device="cuda").long()
        lq = lq.to(self.weight_dtype)
        lq_latent = self.forward_tiled_encode(lq)
        lq_latent = lq_latent.to(self.weight_dtype)
        x_denoised = self.forward_tiled_unet(lq_latent, degra_factor, degra_timesteps, prompts)
        x_denoised = x_denoised.to(self.weight_dtype)
        output_image = self.forward_tiled_decode(x_denoised)
        output_image = output_image.to(self.weight_dtype)

        return output_image

    # @perfcount
    @torch.no_grad()
    def forward_tiled_encode(self, lq):
        # self.save_faeture_map(lq_latent, name='encoder')
        ## add tile function
        t, c, h, w = lq.size()
        tile_size, tile_overlap = (self.args.vae_encoder_tiled_size, self.args.vae_encoder_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[VAE Encoder]: processing......")
            lq_latent_out = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
        else:
            print(f"[VAE Encoder]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            tile_size = min(tile_size, min(h, w))

            # Calculate number of tiles
            num_height_tiles = math.ceil((h - 2 * tile_overlap) / tile_size)
            num_width_tiles = math.ceil((w - 2 * tile_overlap) / tile_size)
            # If any of the numbers are 0, we let it be 1
            # This is to deal with long and thin images
            num_height_tiles = max(num_height_tiles, 1)
            num_width_tiles = max(num_width_tiles, 1)

            # Suggestions from https://github.com/Kahsolt: auto shrink the tile size
            real_tile_height = math.ceil((h - 2 * tile_overlap) / num_height_tiles)
            real_tile_width = math.ceil((w - 2 * tile_overlap) / num_width_tiles)
            real_tile_height = self.get_best_tile_size(real_tile_height, tile_size)
            real_tile_width = self.get_best_tile_size(real_tile_width, tile_size)

            print(f'[VAE Encoder]: split to {num_height_tiles}x{num_width_tiles} = {num_height_tiles*num_width_tiles} tiles. ' +
              f'Optimal tile size {real_tile_width}x{real_tile_height}, original tile size {tile_size}x{tile_size}')

            grid_rows = 0
            cur_x = 0
            while cur_x < lq.size(-1):
                cur_x = max(grid_rows * real_tile_width-tile_overlap * grid_rows, 0)+real_tile_width
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq.size(-2):
                cur_y = max(grid_cols * real_tile_height-tile_overlap * grid_cols, 0)+real_tile_height
                grid_cols += 1

            input_list = []
            lq_latent_list = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * real_tile_width-tile_overlap * row, 0)
                        ofs_y = max(col * real_tile_height-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - real_tile_width
                    if col == grid_cols-1:
                        ofs_y = h - real_tile_height

                    input_start_x = ofs_x
                    input_end_x = ofs_x + real_tile_width
                    input_start_y = ofs_y
                    input_end_y = ofs_y + real_tile_height

                    # input tile dimensions
                    input_tile = lq[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        lq_latent = self.vae.encode(input_list_t).latent_dist.sample() * self.vae.config.scaling_factor
                        input_list = []
                    lq_latent_list.append(lq_latent)

            # Stitch noise predictions for all tiles
            real_tile_height, real_tile_width, tile_overlap = real_tile_height//8, real_tile_width//8, tile_overlap//8
            c, h, w = 4, h//8, w//8
            lq_latent_out = torch.zeros([t, c, h, w], device=lq_latent.device)
            contributors = torch.zeros([t, c, h, w], device=lq_latent.device)
            tile_weights = self._gaussian_weights(real_tile_width, real_tile_height, 1, channels=c)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * real_tile_width-tile_overlap * row, 0)
                        ofs_y = max(col * real_tile_height-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - real_tile_width
                    if col == grid_cols-1:
                        ofs_y = h - real_tile_height

                    input_start_x = ofs_x
                    input_end_x = ofs_x + real_tile_width
                    input_start_y = ofs_y
                    input_end_y = ofs_y + real_tile_height

                    lq_latent_out[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += lq_latent_list[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            lq_latent_out /= contributors

        return lq_latent_out

    # @perfcount
    @torch.no_grad()
    def forward_tiled_decode(self, hq_latent):
        # self.save_faeture_map(lq_latent, name='encoder')
        ## add tile function
        t, c, h, w = hq_latent.size()
        tile_size, tile_overlap = (self.args.vae_decoder_tiled_size, self.args.vae_decoder_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[VAE Decoder]: processing......")
            hq_out = (self.vae.decode(hq_latent / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            print(f"[VAE Decoder]: the input size is {hq_latent.shape[-2]}x{hq_latent.shape[-1]}, need to tiled")
            tile_size = min(tile_size, min(h, w))

            # Calculate number of tiles
            num_height_tiles = math.ceil((h - 2 * tile_overlap) / tile_size)
            num_width_tiles = math.ceil((w - 2 * tile_overlap) / tile_size)
            # If any of the numbers are 0, we let it be 1
            # This is to deal with long and thin images
            num_height_tiles = max(num_height_tiles, 1)
            num_width_tiles = max(num_width_tiles, 1)

            # Suggestions from https://github.com/Kahsolt: auto shrink the tile size
            real_tile_height = math.ceil((h - 2 * tile_overlap) / num_height_tiles)
            real_tile_width = math.ceil((w - 2 * tile_overlap) / num_width_tiles)
            real_tile_height = self.get_best_tile_size(real_tile_height, tile_size)
            real_tile_width = self.get_best_tile_size(real_tile_width, tile_size)

            print(f'[VAE Decoder]: split to {num_height_tiles}x{num_width_tiles} = {num_height_tiles*num_width_tiles} tiles. ' +
              f'Optimal tile size {real_tile_width}x{real_tile_height}, original tile size {tile_size}x{tile_size}')

            grid_rows = 0
            cur_x = 0
            while cur_x < hq_latent.size(-1):
                cur_x = max(grid_rows * real_tile_width-tile_overlap * grid_rows, 0)+real_tile_width
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < hq_latent.size(-2):
                cur_y = max(grid_cols * real_tile_height-tile_overlap * grid_cols, 0)+real_tile_height
                grid_cols += 1

            input_list = []
            hq_list = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * real_tile_width-tile_overlap * row, 0)
                        ofs_y = max(col * real_tile_height-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - real_tile_width
                    if col == grid_cols-1:
                        ofs_y = h - real_tile_height

                    input_start_x = ofs_x
                    input_end_x = ofs_x + real_tile_width
                    input_start_y = ofs_y
                    input_end_y = ofs_y + real_tile_height

                    # input tile dimensions
                    input_tile = hq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        hq = (self.vae.decode(input_list_t / self.vae.config.scaling_factor).sample).clamp(-1, 1)
                        input_list = []
                    hq_list.append(hq)

            # Stitch noise predictions for all tiles
            real_tile_height, real_tile_width, tile_overlap = real_tile_height*8, real_tile_width*8, tile_overlap*8
            c, h, w = 3, h*8, w*8
            hq_out = torch.zeros([t, c, h, w], device=hq.device)
            contributors = torch.zeros([t, c, h, w], device=hq.device)
            tile_weights = self._gaussian_weights(real_tile_width, real_tile_height, 1, channels=c)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * real_tile_width-tile_overlap * row, 0)
                        ofs_y = max(col * real_tile_height-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - real_tile_width
                    if col == grid_cols-1:
                        ofs_y = h - real_tile_height

                    input_start_x = ofs_x
                    input_end_x = ofs_x + real_tile_width
                    input_start_y = ofs_y
                    input_end_y = ofs_y + real_tile_height

                    hq_out[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += hq_list[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            hq_out /= contributors

        return hq_out

    # @perfcount
    @torch.no_grad()
    def forward_tiled_unet(self, lq_latent, degra_factor, degra_timesteps, prompts):
        prompt_embeds = self.encode_prompt(prompts)
        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[Diffusion UNet]: processing......")
            model_pred = self.unet.forward_video_inference(lq_latent, degra_timesteps, encoder_hidden_states=prompt_embeds).sample ## temporal-shift
        else:
            print(f"[Diffusion UNet]: the input size is {lq_latent.shape[-2]}x{lq_latent.shape[-1]}, need to tiled")
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        model_out = self.unet.forward_video_inference(input_list_t, degra_timesteps, encoder_hidden_states=prompt_embeds.to(self.weight_dtype)).sample ## temporal-shift
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = (lq_latent - (1 - degra_factor) ** (0.5) * model_pred) / degra_factor ** (0.5)

        return x_denoised

    def get_best_tile_size(self, lowerbound, upperbound):
        """
        Get the best tile size for GPU memory
        """
        divider = 32
        while divider >= 2:
            remainer = lowerbound % divider
            if remainer == 0:
                return lowerbound
            candidate = lowerbound - remainer + divider
            if candidate <= upperbound:
                return candidate
            divider //= 2
        return lowerbound

    def _gaussian_weights(self, tile_width, tile_height, nbatches, channels=None):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        channels = self.unet.config.in_channels if channels==None else channels
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, channels, 1, 1))

    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n or "RTS" in n:  ## decoder-LoRA   ## temporal-shift
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n or "RTS" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])

