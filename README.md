# FNONet:Boosting Low-Light Image Enhancement by Fourier Neural Operator
## Introduction
This is the official pytorch implementation of "FNONet:Boosting Low-Light Image Enhancement by Fourier Neural Operator" by **Zhongchen**.

Compared to natural scenes with ample illumination, images captured in low-light conditions typically suffer from deficiencies such as underexposure, low contrast, color inaccuracies and limited dynamic range. To enhance the perceptual quality of image in low-light environments, we addresses the limitations of capturing long-range image dependencies in existing methods, assessing the influence of different color space on low-light image enhancement. In this study, we propose FNONet, a two-stage enhancement model that leverages the YCbCr color space and feature fusion.

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



