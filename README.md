<p align="center">
  <img src="./assets/ultravsr_logo.png" height=100>
</p>

# UltraVSR: Achieving Ultra-Realistic Video Super-Resolution with Efficient One-Step Diffusion Space (ACMMM 2025)
[Yong Liu](https://scholar.google.com/citations?user=DT0LPIEAAAAJ&hl=en&oi=sra), 
[Jinshan Pan](https://scholar.google.com/citations?hl=zh-TW&user=CMsNjGIAAAAJ), 
[Yinchuan Li](https://scholar.google.com/citations?hl=zh-TW&user=M6YfuCTSaKsC), 
Qingji Dong, 
Chao Zhu, 
[Yu Guo](https://scholar.google.com/citations?hl=zh-TW&user=OemeiSIAAAAJ), 
[Fei Wang](http://www.aiar.xjtu.edu.cn/info/1046/1242.htm)<br/>


![visitors](https://visitor-badge.laobi.icu/badge?page_id=yongliuy/UltraVSR) 
<a href="https://arxiv.org/abs/2505.19958" target='_blank'><img src="https://img.shields.io/badge/arXiv-2505.19958-b31b1b.svg"></a>
<a href="https://youtu.be/IqH3Y2-4hno" target='_blank'><img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white"></a>
<img alt="GitHub" src="https://img.shields.io/badge/license-Apache_2.0-brightgreen">
[![GitHub Stars](https://img.shields.io/github/stars/yongliuy/UltraVSR?style=social)](https://github.com/yongliuy/UltraVSR/)

:sparkling_heart: If our DITN is helpful to your researches or projects, please help star this repository. Thanks! :hugs: 

<p align="center">
<img src=assets/sampleteaser.png width="1000px"/>
</p>



## :tv: Overview

>In this paper, we propose UltraVSR, a novel framework that enables ultra-realistic and temporal-coherent VSR through an efficient one-step diffusion space. 
A central component of UltraVSR is the Degradation-aware Restoration Schedule (DRS), which estimates a degradation factor from the low-resolution input and transforms iterative denoising process into a single-step reconstruction from low-resolution to high-resolution videos. 
This design eliminates randomness from diffusion noise and significantly speeds up inference. 
To ensure temporal consistency, we propose a lightweight yet effective Recurrent Temporal Shift (RTS) module, composed of an RTS-convolution unit and an RTS-attention unit.
By partially shifting feature components along the temporal dimension, these two units collaboratively facilitate effective feature propagation, fusion, and alignment across neighboring frames, without relying on explicit temporal layers. 
The RTS module is integrated into a pretrained text-to-image diffusion model and is further enhanced through Spatio-temporal Joint Distillation (SJD), which improves temporal coherence while preserving realistic details. 
Additionally, we introduce a Temporally Asynchronous Inference (TAI) strategy to capture long-range temporal dependencies under limited memory constraints. 
Extensive experiments show that UltraVSR achieves state-of-the-art performance, both qualitatively and quantitatively, in a single sampling step. 

<p align="center">
<img src=assets/network.png width="1000px"/>
</p>

## :rocket: Update
- **2025.10**: Inference code is released.
- **2025.07**: Create this repository.

## :mag_right: Dependencies and Installation
1. Clone Repository
    ```bash
    git clone https://github.com/yongliuy/UltraVSR.git
    cd UltraVSR
    ```

2. Create Conda Environment and Install Dependencies
    ```bash
    # create new conda env
    conda create -n ultravsr python=3.10 -y
    conda activate ultravsr

    # install python dependencies
    pip install -r requirements.txt

3. Dependencies:
[Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), 
[TextEncoder](https://drive.google.com/file/d/1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm/view?usp=drive_link), 
[RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth)

## :shopping_cart: Pretrained Models
- Download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1EHdH312K4gwyiU6R_elX0-doXHUx-pLr?usp=sharing).

## :whale: Demo
<p align="center">
  <a href="https://youtu.be/IqH3Y2-4hno">
    <img src="assets/thumbnail.png" width="1000px"/>
  </a>
</p>

## :snowboarder: Running Examples
- Prepare your test images and run the ``inference_UltraVSR.py`` with cuda on command line:
```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python inference_UltraVSR.py \
    --input_video <your_input_video_dataset_path> \
    --output_dir ./results_VideoSR \
    --UltraVSR_weight_path ./pretrained_weights/model_vsr.pkl \
    --upscale 4 \
    --align_method 'wavelet'
```


## :world_map: License
This project is released under the [Apache 2.0 license](./LICENSE). Redistribution and use should follow this license.

## :link: BibTeX
If you find this project useful for your research, please use the following BibTeX entry.
```
@inproceedings{liu2025ultravsr,
  title={UltraVSR: Achieving Ultra-Realistic Video Super-Resolution with Efficient One-Step Diffusion Space},
  author={Liu, Yong and Pan, Jinshan Pan and Li, Yinchuan and Dong, Qingji and Zhu, Chao and Guo, Yu and Wang, Fei},
  booktitle={Proceedings of the 33st ACM International Conference on Multimedia},
  year={2025}
}
```

## :love_letter: Acknowledgments
This project is based on [Diffusers](https://github.com/huggingface/diffusers), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [OSEDiff](https://github.com/cswry/OSEDiff), [RAM](https://drive.google.com/drive/folders/1EHdH312K4gwyiU6R_elX0-doXHUx-pLr?usp=sharing), and [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for their awesome works :star:
