# Introduction

Codes for [RSIPAC track#2](http://rsipac.whu.edu.cn/subject_two). Leave an issue if you have any problems.

# How to run

## Data perparing

Data directory should be like:

```sh
.
└── data
    ├── test
    │   ├── A
    │   ├── B
    │   └── instances_test.json
    └─── train
        ├── A
        ├── B
        └── label
```

## Config

`./configs/base.yaml` is a baseline config, you can refer this for further development.

The first several lines define some global items and reusable items. Like `name` and `version`, these two is used for model log saving. Specifically, model checkpoints and logs will be saved to `./logs/{name}/{version}/*.{ckpt|yaml|csv}`.

The following sections define each component in training:

### 1. `model`

I have implemented some common segmentation models by default, like `segmentation_models_pytorch (smp)` and `mmsegmentation (mmseg)`. The superior model type is defined by `model.type` item in YAML.


#### 1.1 `SMPModel`

For `smp`, I always use its `Unet`, so by default it will give you a `Unet` model. You can change this and other optional arguments in `./src/models/smp_models.py`. 

A reference config is like:

```yaml
model:
  type: SMPModel
  model_type: Unet
  model_name: timm-efficientnet-b0
  pretrained_weight: noisy-student
  num_classes: 1
```


#### 1.2 `MMSegModel`


For `mmseg`, the config defination is almost the same, but I added and deleted some components for my personal perferrence.

I used this instead of original `mmseg` is because it is time-consuming to install `mmseg` on Kaggle platform, so I channged every module that is imported from `mmcv` into pytorch native version. You can easily convert an `mmseg` model file into this repository by changing the import path. 

The converted `mmseg` is at `./src/models/mmseg`. There are already some files, including most frequently used segformer's MiT, `segformer_head` and `uper_head`, but not fully covered the whole `mmseg`.



For example:

```yaml
model:
  type: MMSegModel
  backbone:
    type: timm
    model_name: resnet50
    pretrained: True
    in_chans: 6
  decode_head:
    type: UPerHead
    pool_scales: [1, 2, 3, 6]
    channels: 512
    dropout_ratio: 0.1
    num_classes: 1
    norm_cfg: {type: BN, requires_grad: True}
    align_corners: False
```

Here, I added `timm` backbone for every model that supports [feature extraction](https://rwightman.github.io/pytorch-image-models/feature_extraction/). Arguments excepts `type: timm` will be passed to `timm.create_model()`.

### 2. others

Code is simple, and you can refer the `base.yaml` and `./src/*` for your custum changing.





## Train

```sh
GPUS=0 # or 0,1 for multi-gpu training
python Solver --config /path/to/config --gpus $GPUS
```

## Submission
First, define `names` in `Submission.py`, which is a list of your model names, that is, the `name` item in model config YAML file.

```sh
python Submission.py
```

The results will be saved at `./results/test.segm.json` and `./results.zip`.
