import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import sys
import cv2
import glob
import numpy as np
import pandas as pd
from math import ceil
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
import segmentation_models_pytorch as smp

def rle_decode(mask_rle, shape):
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths
    h, w = shape
    img = np.zeros((h * w,), dtype = np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape)

def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class SMPModel(nn.Module):
    def __init__(self, 
                model_type = "Unet", 
                model_name = "resnet18",
                pretrained_weight = "imagenet",
                num_classes = 1,
                **kwargs
                ):
        super().__init__()
        self.model = getattr(smp, model_type)(model_name, encoder_weights = pretrained_weight, classes = num_classes)
        
    def forward(self, x):
        return self.model(x)
    
def get_model(cfg):
    cfg = cfg.copy()
    model = eval(cfg.pop("type"))(**cfg)
    
    if "load_from" in cfg and cfg.load_from is not None:
        stt = torch.load(cfg.load_from, map_location = "cpu")
        stt = stt["state_dict"] if "state_dict" in stt else stt
        model.load_state_dict(stt, strict = False)
    return model

def load_resize(sub, idx):
    file_name = sub.loc[idx, "file_name"]
    img = cv2.imread(file_name)
    mask = rle_decode(sub.loc[idx, "rle"], img.shape[:2]).T
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    resize_ratio = 1.6 / sub.loc[idx, "pixel_size"] / base_ratio
    old_size = img.shape[:2]
    new_size = (int(round(img.shape[0] / resize_ratio)), int(round(img.shape[1] / resize_ratio)))
    img = cv2.resize(img, new_size[::-1])
    return img, mask, old_size, new_size

def cut_pad_norm(img):
    pad_size = []
    cut_point = []
    for i in range(2):
        h, H, sh = crop_size[i], img.shape[i], stride[i]
        nh = ceil((H - sh) / (h - sh))
        pH = nh * (h - sh) + sh - H
        H += pH
        cut_point.append(np.linspace(0, H - sh, nh, endpoint = False).astype(int))
        pad_size.append([pH // 2, pH - pH // 2])
    pad_img = np.pad(img, [*pad_size, [0,0]])
    pad_img = ((pad_img / 255. - mean) / std).astype(np.float32)
    return pad_img, pad_size, cut_point

def predict(models, cut):
    cut = torch.tensor(cut.transpose(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
        preds = []; sum_weight = 0
        for model, weight, tta in models:
            pred = model(cut).sigmoid()
            for dim in tta:
                pred += torch.flip(model(torch.flip(cut, dim)), dim).sigmoid()
            pred = pred.squeeze().detach().cpu().numpy() / (len(tta) + 1)
            preds.append(pred * weight); sum_weight += weight
        pred = sum(preds) / sum_weight
    return pred

def sliding_window_inference(models, pad_img, pad_size, cut_point, old_size, organ_id = -1):
    base = np.zeros(pad_img.shape[:2], dtype = np.float32)
    sliding_cnt = np.zeros(pad_img.shape[:2], dtype = int)
    for i, x in enumerate(cut_point[0]):
        for j, y in enumerate(cut_point[1]):
            cut = pad_img[x:x + crop_size[0], y:y + crop_size[1]]
            
            pred = predict(models, cut)
            if organ_id != -1: pred = pred[organ_id]
            
            base[x:x + crop_size[0], y:y + crop_size[1]] += pred
            sliding_cnt[x:x + crop_size[0], y:y + crop_size[1]] += 1 # kernel
    base /= sliding_cnt
    base = base[pad_size[0][0]:base.shape[0]-pad_size[0][1], pad_size[1][0]:base.shape[1]-pad_size[1][1]]
    base = cv2.resize(base, old_size[::-1])
    return base



def iter_folds():
    D = []
    for fold in range(5):
        data_dir = "./data"
        sub = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_images = glob.glob(os.path.join(data_dir, "train/images", "*.*"), recursive = True)
        sub["uid"] = [f"{o}_{i}" for o, i in zip(sub.organ, sub.id)]

        sub = sub[sub.uid.isin(pd.read_csv(os.path.join(data_dir, f"./train/splits/holdout_{fold}.txt"), header = None).iloc[:,0])].reset_index(drop = True)

            
        id2img = {int(os.path.basename(_).split(".")[0]): _ for _ in test_images}
        sub["file_name"] = sub.id.map(id2img)


        model_infos = [
            dict(
                ckpt = f"./logs/{name}/f{fold}/{last_or_best}.ckpt",
                weight = 1.0,
                tta = [[2], [3], [2, 3]]
            )
        ]

        models = []
        for model_info in model_infos:
            if not os.path.exists(model_info["ckpt"]):
                model_info['ckpt'] = glob.glob(model_info['ckpt'])[0]
            stt = torch.load(model_info["ckpt"], map_location = "cpu")
            cfg = OmegaConf.create(eval(str(stt["hyper_parameters"]))).model
            cfg.pretrained_weight = None
            stt = {k[6:]: v for k, v in stt["state_dict"].items()}
            
            model = get_model(cfg)
            model.load_state_dict(stt)
            model.eval()
            model.cuda()
            models.append([model, model_info["weight"], model_info["tta"]])


        dices = []
        for idx in tqdm(range(len(sub))):
            organ = sub.loc[idx, "organ"]
            img, mask, old_size, new_size = load_resize(sub, idx)
            
            pad_img, pad_size, cut_point = cut_pad_norm(img)
            
            if organ in organ_id:
                pred = sliding_window_inference(models, pad_img, pad_size, cut_point, old_size, organ_id[organ])
                pred = (pred > thres.get(organ, 0.5)).astype(np.uint8)
            else:
                pred = np.ones(old_size, dtype = np.uint8)
            
            I = ((pred == 1) & (mask == 1)).sum()
            C = (pred == 1).sum() + (mask == 1).sum()
            dices.append([2 * I / C, organ])
        dices = pd.DataFrame(dices, columns = ["dice", "organ"])
        D.append(dices.groupby("organ").agg("mean").T)
    return D

base_ratio = 1.0
crop_size = tuple([int(round(_ * base_ratio)) for _ in (512, 512)])
stride = tuple([int(round(_ * base_ratio)) for _ in (256, 256)])
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
thres = {}
organs = ["prostate", "spleen", "largeintestine", "kidney", "lung"]
organ_id = {o: -1 for i, o in enumerate(organs)}

name = "base_b2"
last_or_best = "last" # ["epoch*", "last"]
D = iter_folds()
d = pd.concat(D).reset_index(drop = True)
d