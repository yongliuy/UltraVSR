import os, sys
sys.path.append(os.getcwd())
from models.autoencoder_kl import AutoencoderKL
from models.autoencoder_kl_with_RTS import CustomVidAutoencoderKL
from models.unet_2d_with_RTS import UNet2DConditionVideoModel
from peft import LoraConfig

def initialize_vae(args, with_RTS=False):
    if with_RTS:
        vae = CustomVidAutoencoderKL.from_pretrained(args.pretrained_diffusion_model_path, subfolder="vae", low_cpu_mem_usage=False)
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_diffusion_model_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules, l_target_modules_shift = [], []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "RTS" in n and with_RTS:
            l_target_modules_shift.append(n)
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n) and ("RTS" not in n):
            # if pattern in n: ## decoder-LoRA
                l_target_modules.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n) and ("RTS" not in n):
            # elif 'quant_conv' in n: ## decoder-LoRA
                l_target_modules.append(n.replace(".weight",""))
    
    lora_conf = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules)
    vae.add_adapter(lora_conf, adapter_name="default_encoder")

    if with_RTS:
        return vae, l_target_modules, l_target_modules_shift
    else:
        return vae, l_target_modules

def initialize_unet(args, with_RTS=False, with_LoRA=True):
    unet = UNet2DConditionVideoModel.from_pretrained(args.pretrained_diffusion_model_path, subfolder="unet", low_cpu_mem_usage=False)
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others, l_modules_shift = [], [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "RTS" in n and with_RTS:
            l_modules_shift.append(n)
        if with_LoRA:
            if "bias" in n or "norm" in n:
                continue
            for pattern in l_grep:
                if pattern in n and ("down_blocks" in n or "conv_in" in n) and ("RTS" not in n):
                    l_target_modules_encoder.append(n.replace(".weight",""))
                    break
                elif pattern in n and ("up_blocks" in n or "conv_out" in n) and ("RTS" not in n):
                    l_target_modules_decoder.append(n.replace(".weight",""))
                    break
                elif pattern in n and ("RTS" not in n):
                    l_modules_others.append(n.replace(".weight",""))
                    break

    if with_LoRA:
        lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
        lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
        lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
        unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        unet.add_adapter(lora_conf_others, adapter_name="default_others")

    if with_RTS:
        if with_LoRA:
            return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others, l_modules_shift
        else:
            return unet, l_modules_shift
    else:
        if with_LoRA:
            return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
        else:
            return unet