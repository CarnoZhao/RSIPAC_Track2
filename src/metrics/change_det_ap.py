import torch
import cv2
import numpy as np
import pycocotools.mask as mutils
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from contextlib import redirect_stdout


class ChangeAP(object):
    def __init__(self, thres_list = [0.2]):
        self.thres_list = thres_list
        self.clean()


    def clean(self):
        self.gts = {"images": [], "annotations": [], "categories": []}
        self.gts["categories"] = [{"name": "change", "id": 1}]
        self.dts = {thres: [] for thres in self.thres_list}

    def get_gt(self, mask):
        nc, label = cv2.connectedComponents(mask, connectivity = 8)
        self.gts["images"].append({
            "file_name": "",
            "width": mask.shape[1],
            "height": mask.shape[0],
            "id": len(self.gts["images"])
        })
        for c in range(nc):
            if np.all(mask[label == c] == 0):
                continue
            else:
                ann = np.asfortranarray((label == c).astype(np.uint8))
                rle = mutils.encode((ann))
                bbox = [int(_) for _ in mutils.toBbox(rle)]
                area = bbox[-1] * bbox[-2]
                self.gts["annotations"].append({
                    "segmentation": {
                        "size": [int(_) for _ in rle["size"]], 
                        "counts": rle["counts"].decode()},
                    "bbox": bbox, "area": area, "iscrowd": 0, "category_id": 1,
                    "image_id": len(self.gts["images"]) - 1, "id": len(self.gts["annotations"])
                })
    
    def get_dt(self, pred, thres):
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
                score = float(np.max(pred[label == c]))
                self.dts[thres].append({
                    "segmentation": {
                        "size": [int(_) for _ in rle["size"]], 
                        "counts": rle["counts"].decode()},
                    "bbox": bbox, "area": area, "iscrowd": 0, "category_id": 1,
                    "image_id": len(self.gts["images"]) - 1, "id": len(self.dts[thres]),
                    "score": score
                })

    def preprocess(self, output):
        y, yhat = output
        y = y.detach().cpu().numpy().astype(np.uint8)
        yhat = yhat.sigmoid().squeeze(1).detach().cpu().numpy()
        for mask, pred in zip(y, yhat):
            self.get_gt(mask)
            for thres in self.dts:
                self.get_dt(pred, thres)

    def __call__(self, outputs):
        with open("/tmp/gt.json", "w") as f:
            json.dump(self.gts, f)

        aps = []
        with open('/dev/null', 'a') as f:
            with redirect_stdout(f):
                for thres in self.dts:
                    with open("/tmp/dt.json", "w") as f:
                        json.dump(self.dts[thres], f)
                    gt = COCO("/tmp/gt.json")
                    dt = gt.loadRes("/tmp/dt.json")
                    e = COCOeval(gt, dt, "segm")
                    e.params.iouThrs = [0.1]
                    e.evaluate()
                    e.accumulate()
                    e.summarize()
                    aps.append(e.stats[0])

        res = {}
        for thres, ap in zip(self.dts, aps):
            res[f"val_ap_{thres:.2f}"] = ap
        res["val_ap"] = max(aps)
        res["best_thres"] = np.array(list(self.dts.keys()))[np.argmax(aps)]
        self.clean()
        
        return res

