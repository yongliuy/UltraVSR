import os
import cv2
import random
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from dataloaders.realesrgan import RealESRGAN_degradation
# from realesrgan import RealESRGAN_degradation

class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution, args.resolution)),
                transforms.RandomHorizontalFlip(),
            ])

            self.gt_list = []
            for gt_path in args.dataset_paths_list:
                print(gt_path)
                for root, dirs, files in os.walk(gt_path):
                    for name in files:
                        if name.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
                            self.gt_list.append(os.path.join(root, name))

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            img_width, img_height = gt_img.size
            while img_width<self.args.resolution or img_height<self.args.resolution:
                idx = np.random.randint(len(self.gt_list))
                gt_img = Image.open(self.gt_list[idx]).convert('RGB')
                img_width, img_height = gt_img.size
            gt_img = self.crop_preproc(gt_img)

            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example

def find_folders_with_files(path, extensions, subfolder=False):
    folders = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                if subfolder:
                    path = os.path.join(os.path.abspath(root), file)
                    if path not in folders:
                        folders.append(os.path.join(os.path.abspath(root), file))
                else:
                    if root not in folders:
                        folders.append(root)
    
    return folders

class PairedSROnlineVidDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            # self.crop_preproc = transforms.Compose([
            #     transforms.RandomCrop((args.resolution, args.resolution)),
            #     transforms.RandomHorizontalFlip(),
            # ])
            # self.clip_transform = transforms.RandomHorizontalFlip()

            self.gt_list = []
            for gt_path in args.dataset_paths_list:
                print(gt_path)
                if "REDS" in gt_path:
                    gt_list = find_folders_with_files(gt_path, ['.png'], subfolder=False)
                    self.gt_list += gt_list*40
                if "YouHQ" in gt_path:
                    gt_list = find_folders_with_files(gt_path, ['.mp4'], subfolder=True)
                    # random.shuffle(gt_list)
                    self.gt_list += gt_list

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            gts = self.gt_list[idx]
            if gts.endswith('.mp4'):
                ## read video
                video = cv2.VideoCapture(gts)
                data_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                start_frame_idx = random.randint(0, data_frames-self.args.train_sequence_length)
                end_frame_idx = start_frame_idx+self.args.train_sequence_length
                frames = []
                video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
                for _ in range(start_frame_idx, end_frame_idx):
                    ret, frame = video.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                video.release()
            else:
                ## read image sequence
                imgs = os.listdir(gts)
                imgs.sort()
                data_frames = len(imgs)
                start_frame_idx = random.randint(0, data_frames-self.args.train_sequence_length)
                end_frame_idx = start_frame_idx+self.args.train_sequence_length
                frames = []
                for i in range(start_frame_idx, end_frame_idx):
                    frame = cv2.imread(os.path.join(gts, imgs[i]))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            processed_frames = []
            is_first_frame = True
            is_horizontal_flip = random.random() < 0.5
            for frame in frames:
                pil_img = transforms.ToPILImage()(frame)
                if is_first_frame:
                    i,j,h,w = transforms.RandomCrop.get_params(pil_img,output_size=(self.args.resolution, self.args.resolution))
                    is_first_frame = False
                pil_img = transforms.functional.crop(pil_img,i,j,h,w)
                if is_horizontal_flip:
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                processed_frame = np.array(pil_img)
                processed_frames.append(processed_frame)

            processed_frames = np.concatenate(processed_frames, axis=1)
            output_t, img_t = self.degradation.degrade_process(np.asarray(processed_frames)/255., resize_bak=True)
            output_t = list(torch.split(output_t, self.args.resolution, dim=-1))
            img_t = list(torch.split(img_t, self.args.resolution, dim=-1))
            output_t = torch.cat(output_t, dim=0)
            img_t = torch.cat(img_t, dim=0)

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = [self.args.neg_prompt]*self.args.train_sequence_length
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    ## '/home/zhuchao/final/dataset/REDS/train_gt', '../../VSR/datasets/YouHQ-Train'
    parser.add_argument("--dataset_paths_list", type=str, nargs='+', default=['/home/zhuchao/final/dataset/REDS/train_gt'])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_sequence_length", type=int, default=5)
    parser.add_argument("--seq_length", type=int, default=5)
    parser.add_argument("--neg_prompt", type=str, default="painting, oil painting, illustration, drawing")
    parser.add_argument("--deg_file_path", default="params_realesrgan.yml", type=str)
    args = parser.parse_args()

    dataset_train = PairedSROnlineVidDataset(split="train", args=args)
    print(len(dataset_train))
    example = dataset_train[100]
    lr, gt = example["conditioning_pixel_values"], example["output_pixel_values"]

    print(example["output_pixel_values"].shape)
    print(example["conditioning_pixel_values"].shape)

    for i in range(5):
        output_pil = transforms.ToPILImage()(lr[i].cpu() * 0.5 + 0.5)
        output_pil.save(f'lr100_{i}.png')
        output_pil = transforms.ToPILImage()(gt[i].cpu() * 0.5 + 0.5)
        output_pil.save(f'gt100_{i}.png')