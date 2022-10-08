import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import sys
import cv2
import glob
import json
import numpy as np
import pandas as pd
from math import ceil
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval 
import pycocotools.mask as mutils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf

from src.models import models as Registry

data_dir = "./data/test"
train_images_A = sorted(glob.glob(os.path.join(data_dir, "A/*")))
train_images_B = sorted(glob.glob(os.path.join(data_dir, "B/*")))
splits = sorted(glob.glob(os.path.join(data_dir, "splits/fold_*")))
df = pd.DataFrame({"image_file_A": train_images_A, "image_file_B": train_images_B})
df["uid"] = df.image_file_A.apply(lambda x: int(os.path.basename(x).split(".")[0]))


def get_model(cfg):
    cfg = cfg.copy()
    model = Registry[cfg.pop("type")](**cfg)
    return model

def get_models(names, folds):
    model_infos = [
        dict(
            ckpt = f"./logs/{name}/f{fold}/{last_or_best}.ckpt",
            weight = weight,
            tta = tta,
            exclude_func = exclude_func
        ) for name, weight, exclude_func in names for fold in folds
    ]
    models = []
    for model_info in model_infos:
        if not os.path.exists(model_info["ckpt"]):
            model_info['ckpt'] = sorted(glob.glob(model_info['ckpt']))[-1]
        stt = torch.load(model_info["ckpt"], map_location = "cpu")
        cfg = OmegaConf.create(eval(str(stt["hyper_parameters"]))).model
        if cfg.type == "Segformer":
            cfg.pretrained = True
        elif cfg.type == "MMSegModel":
            pass
        elif cfg.type == "SMPModel":
            cfg.pretrained_weight = None
        stt = {k[6:]: v for k, v in stt["state_dict"].items()}

        model = get_model(cfg)
        model.load_state_dict(stt, strict = True)
        model.eval()
        model.cuda()
        models.append([model, 
                       model_info["weight"], 
                       model_info["tta"],
                       model_info["exclude_func"]])
    return models

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
def load(row):
    imgA = cv2.imread(row.image_file_A)
    imgB = cv2.imread(row.image_file_B)
    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
    imgA = (imgA / 255. - mean) / std
    imgB = (imgB / 255. - mean) / std
    if siamese:
        img = (imgA.astype(np.float32), imgB.astype(np.float32))
    else:
        img = np.concatenate([imgA, imgB], -1).astype(np.float32)
    return img, None


def predict(row, models, img):
    if siamese:
        img = (torch.tensor(img[0].transpose(2, 0, 1)).unsqueeze(0).cuda(),
               torch.tensor(img[1].transpose(2, 0, 1)).unsqueeze(0).cuda())
    else:
        img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
        preds = []; sum_weight = 0
        for model, weight, tta, exclude_func in models:
            if exclude_func is not None and exclude_func(row): continue
            pred = model(img).sigmoid()
            for dim in tta:
                if siamese:
                    pred += torch.flip(model(
                        (torch.flip(img[0], dim), torch.flip(img[1], dim))
                    ), dim).sigmoid()
                else:
                    pred += torch.flip(model(torch.flip(img, dim)), dim).sigmoid()
            pred = pred.squeeze().detach().cpu().numpy() / (len(tta) + 1)
            preds.append(pred * weight); sum_weight += weight
        pred = sum(preds) / sum_weight
    return pred


def get_dt(row, pred, img_id, dts, thres):
    mask = (pred > thres).astype(np.uint8)
    nc, label = cv2.connectedComponents(mask, connectivity = 8)
    for c in range(nc):
        if np.all(mask[label == c] == 0):
            continue
        else:
            ann = np.asfortranarray((label == c).astype(np.uint8))
            rle = mutils.encode((ann))
            bbox = [int(_) for _ in mutils.toBbox(rle)]
            area = int(mutils.area(rle))
            score = float(pred[label == c].max())
            dts.append({
                "segmentation": {
                    "size": [int(_) for _ in rle["size"]], 
                    "counts": rle["counts"].decode()},
                "bbox": [int(_) for _ in bbox], "area": int(area), "iscrowd": 0, "category_id": 1,
                "image_id": int(img_id), "id": len(dts),
                "score": float(score)
            })

siamese = True
names = [
    ["sia_pv2d", 1.0, None],
]
tta = [[2], [3], [2,3]]
last_or_best = ["epoch*", "last"][0]
thres = 0.2
folds = list(range(5))



os.system("mkdir -p results")
sub = df
models = get_models(names, folds)
dts = []
for idx in tqdm(range(len(sub))):
    row = sub.loc[idx]
    img, mask = load(row)
    pred = predict(row, models, img)
    get_dt(row, pred, row.uid, dts, thres)
with open("./results/test.segm.json", "w") as f:
    json.dump(dts, f)
os.system("zip -9 -r results.zip results")
