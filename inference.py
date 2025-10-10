import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
from PIL import Image
from models.ram.models.ram_lora import ram
from models.ram import inference_ram as inference
from models.inference_arch import UltraVSR_test
from wavelet_color_fix import adain_color_fix, wavelet_color_fix

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, model)
    validation_prompt = f"{captions[0]}"
    
    return validation_prompt, lq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', '-i', type=str, default='./Cloud_VideoSR', help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, default='./results_Cloud_VideoSR', help='the directory to save the output')
    parser.add_argument("--UltraVSR_weight_path", type=str, default='./pretrained_weights/model_UltraVSR.pkl')
    parser.add_argument('--pretrained_diffusion_model_path', type=str, default='./weights/stable-diffusion-2-1-base/', help='sd model path')
    parser.add_argument('--ram_path', type=str, default='./weights/ram_swin_large_14m.pth')
    parser.add_argument('--ram_ft_path', type=str, default='./weights/DAPE.pth')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--max_process_size", type=int, default=2500) # Modify according to the available memory
    parser.add_argument("--max_process_size_per_interval", type=int, default=1440) # Modify according to the available memory
    parser.add_argument("--max_process_interval", type=int, default=30) # Modify according to the available memory
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='wavelet')
    parser.add_argument('--resize_back', type=bool, default=True)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False)
    # tile setting - Encoder - Based on memory size
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=3000) # Modify according to the available memory
    parser.add_argument("--vae_encoder_overlap", type=int, default=32) # Modify according to the available memory
    parser.add_argument("--vae_encoder_num_frames_per_batch", type=int, default=1) # Modify according to the available memory
    # tile setting - Decoder - Based on memory size
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=96)  # Modify according to the available memory
    parser.add_argument("--vae_decoder_overlap", type=int, default=10)  # Modify according to the available memory
    parser.add_argument("--vae_decoder_num_frames_per_batch", type=int, default=1)  # Modify according to the available memory
    # tile setting - Latent - Based on memory size
    parser.add_argument("--latent_tiled_size", type=int, default=96) # Modify according to the available memory
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)  # Modify according to the available memory
    parser.add_argument("--latent_num_frames_per_batch", type=int, default=1) # Modify according to the available memory
    
    args = parser.parse_args()

    # initialize the model
    model = UltraVSR_test(args)

    # get ram model
    DAPE = ram(pretrained=args.ram_path,
            pretrained_condition=args.ram_ft_path,
            image_size=384,
            vit='swin_l')
    DAPE.eval()
    DAPE.to("cuda")

    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # set weight type
    DAPE = DAPE.to(dtype=weight_dtype)

    # get all input images
    sequence_paths = sorted(os.listdir(args.input_video))
    for sequence_path in sequence_paths:
        image_names = sorted(glob.glob(f'{os.path.join(args.input_video, sequence_path)}/*.png'))

        # make the output dir
        os.makedirs(os.path.join(args.output_dir, sequence_path), exist_ok=True)
        print(f'There are {len(image_names)} images.')
        lqs, input_images, validation_prompts, bnames = [], [], [], []
        for image_name in image_names:
            # make sure that the input image is a multiple of 8
            input_image = Image.open(image_name).convert('RGB')
            ori_width, ori_height = input_image.size
            if ori_width < args.process_size//args.upscale or ori_height < args.process_size//args.upscale:
                scale = (args.process_size)/min(ori_width, ori_height)
                print(f"upsample {image_name} to {int(scale*ori_width)}x{int(scale*ori_height)}")
                input_image = input_image.resize((int(scale*ori_width), int(scale*ori_height)))
                resize_flag = True
            elif ori_width > args.max_process_size//args.upscale or ori_height > args.max_process_size//args.upscale:
                scale = (args.max_process_size)/max(ori_width, ori_height)
                print(f"downsample {image_name} to {int(scale*ori_width)}x{int(scale*ori_height)}")
                input_image = input_image.resize((int(scale*ori_width), int(scale*ori_height)))
                resize_flag = True
            else:
                input_image = input_image.resize((ori_width*args.upscale, ori_height*args.upscale))
                resize_flag = False

            new_width = input_image.width - input_image.width % 8
            new_height = input_image.height - input_image.height % 8
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
            bname = os.path.basename(image_name)
            input_images.append(input_image)
            bnames.append(bname)

            # get caption
            validation_prompt, lq = get_validation_prompt(args, input_image, DAPE)
            validation_prompts.append(validation_prompt)
            lq = lq*2-1
            lqs.append(lq)
            # print(f"process {image_name}, tag: {validation_prompt}".encode('utf-8'))

        if len(lqs) % args.latent_num_frames_per_batch > 0:
            res_num = args.latent_num_frames_per_batch - len(lqs) % args.latent_num_frames_per_batch
            print(f"The length of the input video sequence cannot be divided by latent_num_frames_per_batch and needs to be expanded, res_num: {res_num}")
            for i in range(res_num):
                lqs.append(lqs[-1])
                validation_prompts.append(validation_prompts[-1])

        lqs = torch.cat(lqs, 0)
        ## max_process_interval
        tn, _, th, tw = lqs.shape
        indices = torch.linspace(0, tn - 1, steps=tn//10).long()
        lqs_sampled = lqs[indices][:5]
        degra_factor = model.metric_model(lqs_sampled*0.5+0.5)
        degra_factor = torch.mean(degra_factor)**2
        
        if (th > args.max_process_size_per_interval or tw > args.max_process_size_per_interval) and tn > args.max_process_interval:
            lqs = [lqs[i:min(len(lqs), i + args.max_process_interval)] for i in range(0, len(lqs), args.max_process_interval)]
            print(f"Due to the high resolution, the input video sequence is split into multiple intervals to prevent out-of-memory errors, max_process_interval: {args.max_process_interval}. You can modify this setting according to the available memory.")
        # torch.cuda.empty_cache()

        # translate the image
        with torch.no_grad():
            if type(lqs) == list:
                img_index = 0
                for i in range(len(lqs)):
                    print(f"Processing: {i+1}/{len(lqs)}")
                    output_images = model(lqs[i], degra_factor, prompts=validation_prompts)

                    for j in range(len(output_images)):
                        output_pil = transforms.ToPILImage()(output_images[j].cpu() * 0.5 + 0.5)
                        if args.align_method == 'adain':
                            output_pil = adain_color_fix(target=output_pil, source=input_images[img_index])
                        elif args.align_method == 'wavelet':
                            output_pil = wavelet_color_fix(target=output_pil, source=input_images[img_index])
                        else:
                            pass
                        if args.resize_back and resize_flag:
                            output_pil.resize((int(args.upscale*ori_width), int(args.upscale*ori_height)))
                        output_pil.save(os.path.join(args.output_dir, sequence_path, bnames[img_index]))
                        img_index += 1
            else:
                output_images = model(lqs, degra_factor, prompts=validation_prompts)
                for i in range(len(output_images)):
                    output_pil = transforms.ToPILImage()(output_images[i].cpu() * 0.5 + 0.5)
                    if args.align_method == 'adain':
                        output_pil = adain_color_fix(target=output_pil, source=input_images[i])
                    elif args.align_method == 'wavelet':
                        output_pil = wavelet_color_fix(target=output_pil, source=input_images[i])
                    else:
                        pass
                    if args.resize_back and resize_flag:
                        output_pil.resize((int(args.upscale*ori_width), int(args.upscale*ori_height)))
                

                    output_pil.save(os.path.join(args.output_dir, sequence_path, bnames[i]))
            torch.cuda.empty_cache()
    print('All videos have been successfully processed.')