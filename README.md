# FNONet:Boosting Low-Light Image Enhancement by Fourier Neural Operator
## Introduction
This is the official pytorch implementation of "FNONet:Boosting Low-Light Image Enhancement by Fourier Neural Operator" by **Zhongchen**.

Compared to natural scenes with ample illumination, images captured in low-light conditions typically suffer from deficiencies such as underexposure, low contrast, color inaccuracies and limited dynamic range. To enhance the perceptual quality of image in low-light environments, we addresses the limitations of capturing long-range image dependencies in existing methods, assessing the influence of different color space on low-light image enhancement. In this study, we propose FNONet, a two-stage enhancement model that leverages the YCbCr color space and feature fusion.

![image](https://github.com/Zhong-Chenchen/FNONet/assets/93313310/daa55101-5367-4845-b57d-78eff0048cf6)

Initially, RGB images undergo a transformation to the YCbCr space, with spectral mappings elicited via bidimensional Fourier neural operators, thereby elevating luminance levels. Subsequently, a spatio-frequency fusion module is integrated, amalgamating global and local attributes to reinstate image detail. The proposed FNONet framework reconciles efficiency with performance, optimizing color space processing and perceptually enhancing low-quality images.

Experiments on three canonical datasets: LOL-v1, LOL-v2-real, and LOL-v2-synthetic, demonstrate that the proposed FNONet achieves significant performance in low-light enhancement, outperforming current mainstream LLIE methods overall. FNONet successfully recovers fine details and textures in the image restoration process, confirming its effectiveness in restoring low-light image quality. Future research will explore the extension of the Fourier neural operator to three-dimensional spaces, achieving integrated modeling of the YCbCr three color channels, to further enhance the model's generalization capability.
## Installation

```
conda create --name FourLLIE --file requirements.txt
conda activate FourLLIE
```

## Train

You can modify the training configuration (e.g., the path of datasets, learning rate, or model settings) in `./options/train/LOLv2_real.yml` and run:

```
python train.py -opt ./options/train/LOLv2_real.yml
```

## Test

Modify the testing configuration in `./options/test/LOLv2_real.yml` and run:

```
python test.py -opt ./options/test/LOLv2_real.yml
```



