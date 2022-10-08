import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import cv2
import time
import glob
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mutils
from contextlib import redirect_stdout

import torch
import torch.nn as nn

from omegaconf import OmegaConf
sys.path.append("/home/zhaoxun/codes/Rsipac")
from src.models import models as Registry

data_dir = "../data/train"
train_images_A = sorted(glob.glob(os.path.join(data_dir, "A/*")))
train_images_B = sorted(glob.glob(os.path.join(data_dir, "B/*")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "label/*")))
splits = sorted(glob.glob(os.path.join(data_dir, "splits/fold_*")))
df = pd.DataFrame({"image_file_A": train_images_A, "image_file_B": train_images_B, "mask_file": train_labels})
df["uid"] = df.image_file_A.apply(lambda x: int(os.path.basename(x).split(".")[0]))
df["fold"] = -1
for split in splits:
    fold = int(os.path.basename(split).split(".")[0].split("_")[1])
    split = pd.read_csv(split, header = None)
    df.loc[df.uid.isin(split.iloc[:,0]), "fold"] = fold

class SiameseModelTTA(nn.Module):
    def __init__(self, model):
        super(SiameseModelTTA, self).__init__()
        self.model = model.half()
        self.act = nn.Sigmoid()
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).half())
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).half())

    def forward(self, img):
        with torch.cuda.amp.autocast(True):
            img1 = (torch.flip(img[0], dims = [2]).half() / 255. - self.mean) / self.std
            img2 = (torch.flip(img[1], dims = [2]).half() / 255. - self.mean) / self.std
            img1 = img1.permute(2, 0, 1)
            img2 = img2.permute(2, 0, 1)
            img1 = torch.stack([
                img1,
                torch.flip(img1, dims = [1]),
                torch.flip(img1, dims = [2]),
                torch.flip(img1, dims = [1, 2]),
            ], 0)
            img2 = torch.stack([
                img2,
                torch.flip(img2, dims = [1]),
                torch.flip(img2, dims = [2]),
                torch.flip(img2, dims = [1, 2]),
            ], 0)
            out = self.model((img1, img2))
            out = out.float()
            out = self.act(out)
            ret = out[0]
            ret += torch.flip(out[1], dims = [1])
            ret += torch.flip(out[2], dims = [2])
            ret += torch.flip(out[3], dims = [1, 2])
            return ret / 4

def get_model(cfg):
    cfg = cfg.copy()
    model = Registry[cfg.pop("type")](**cfg)
    return model

def get_models(names, fold):
    model_infos = [
        dict(
            ckpt = f"../logs/{name}/f{fold}/{last_or_best}.ckpt",
            weight = weight,
            exclude_func = exclude_func
        ) for name, weight, exclude_func in names
    ]
    models = []
    for model_info in model_infos:
        if not os.path.exists(model_info["ckpt"]):
            model_info['ckpt'] = sorted(glob.glob(model_info['ckpt']))[-1]
        stt = torch.load(model_info["ckpt"], map_location = "cpu")
        cfg = OmegaConf.create(eval(str(stt["hyper_parameters"]))).model
        if cfg.type == "Segformer":
            cfg.pretrained = True
        elif cfg.type.startswith("MM"):
            cfg.backbone.pretrained = None
        elif cfg.type == "SMPModel":
            cfg.pretrained_weight = None
        stt = {k[6:]: v for k, v in stt["state_dict"].items()}

        model = get_model(cfg)
        model.load_state_dict(stt, strict = True)
        model = SiameseModelTTA(model)
        model.eval()
        model.cuda()
        models.append([model, 
                       model_info["weight"], 
                       model_info["exclude_func"]])
    return models


def load(row):
    imgA = cv2.imread(row.image_file_A)
    imgB = cv2.imread(row.image_file_B)
    
    return (imgA, imgB), None # mask


def predict(row, models, img):
    img = [torch.tensor(_).cuda() for _ in img]
    with torch.no_grad():
        preds = []; sum_weight = 0
        for model, weight, exclude_func in models:
            if exclude_func is not None and exclude_func(row): continue
            pred = model(img)
            pred = pred.squeeze().detach().cpu().numpy()
            preds.append(pred * weight); sum_weight += weight
        pred = sum(preds) / sum_weight
    return pred

def get_dt(row, pred, img_id, dts, thres):
    mask = (pred > thres).astype(np.uint8)
    nc, label, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity = 8)
    for c in range(nc):
        if np.all(mask[label == c] == 0):
            continue
        else:
            ann = np.asfortranarray((label == c).astype(np.uint8))
            rle = mutils.encode((ann))
            bbox = [int(_) for _ in mutils.toBbox(rle)]
            area = int(stats[c,4])
            score = float(np.max(pred[label == c]))
            dts.append({
                "segmentation": {
                    "size": [int(_) for _ in rle["size"]], 
                    "counts": rle["counts"].decode()},
                "bbox": bbox, "area": area, "iscrowd": 0, "category_id": 1,
                "image_id": img_id, "id": len(dts),
                "score": score
            })


names = [
    ["sia_pv2d", 1.0, None]
]
tta = [[2], [3], [2,3]]
last_or_best = ["epoch*", "last"][0]
thres = 0.2
fold = 0


sub = df[df.fold == fold].copy().reset_index(drop = True)
models = get_models(names, fold)

total_time = 0
total_time -= time.time()
dts = []
for idx in tqdm(range(len(sub))):
    row = sub.loc[idx]
    img, mask = load(row)
    pred = predict(row, models, img)
    get_dt(row, pred, idx, dts, thres)
total_time += time.time()

with open("/tmp/dt.json", "w") as f:
    json.dump(dts, f)

with open('/dev/null', 'a') as f:
    with redirect_stdout(f):
        gt = COCO(f"../data/train/annotations/holdout_{fold}.json")
        dt = gt.loadRes("/tmp/dt.json")
        e = COCOeval(gt, dt, "segm")
        e.params.iouThrs = [0.1]
        e.evaluate()
        e.accumulate()
e.summarize()
print(fold, f"{total_time:.3f}", e.stats[0])
