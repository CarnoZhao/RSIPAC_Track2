import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from .seg_dataset import SegData
from .builder import build_trans

class ConcatData(SegData):
    def __init__(self, preload = False, shuffle_ratio = 0.0, color_trans = None, **kwargs):
        super().__init__(**kwargs)
        if color_trans is not None:
            self.color_trans = build_trans(
                color_trans.get(self.phase, None), 
                T = color_trans.get("type", "A"),
                trans_args = color_trans.get("trans_args", {}))
        else:
            self.color_trans = None
        self.shuffle_ratio = shuffle_ratio
        self.cache = []
        if preload:
            for idx in range(len(self.df)):
                self.cache.append(self.load_from_df(idx))  

    def load_from_df(self, idx):
        row = self.df.loc[idx]
        if row["redirect"] != -1:
            idx = np.random.randint(len(self.value_df[row["redirect"]]))
            row = self.value_df[row["redirect"]].loc[idx]

        image_file_A = row["image_file_A"]
        image_file_B = row["image_file_B"]
        mask_file = row["mask_file"]
        imgA = cv2.imread(image_file_A)
        imgB = cv2.imread(image_file_B)
        imgA, imgB = [cv2.cvtColor(_, cv2.COLOR_BGR2RGB) for _ in (imgA, imgB)]
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)  
        return imgA, imgB, mask   

    @staticmethod
    def prepare(data_dir = "./data/train", **dataset_cfg):
        train_images_A = sorted(glob.glob(os.path.join(data_dir, "A/*")))
        train_images_B = sorted(glob.glob(os.path.join(data_dir, "B/*")))
        train_labels = sorted(glob.glob(os.path.join(data_dir, "label/*")))
        df = pd.DataFrame({"image_file_A": train_images_A, "image_file_B": train_images_B, "mask_file": train_labels})
        df["uid"] = df.image_file_A.apply(lambda x: int(os.path.basename(x).split(".")[0]))
        splits = sorted(glob.glob(os.path.join(data_dir, "splits/fold_*")))
        if splits:
            df["fold"] = -1
            for split in splits:
                fold = int(os.path.basename(split).split(".")[0].split("_")[1])
                split = pd.read_csv(split, header = None)
                df.loc[df.uid.isin(split.iloc[:,0]), "fold"] = fold
        return df

    def prepare_item(self, idx):
        if self.cache:
            imgA, imgB, mask = self.cache[idx]
        else:
            imgA, imgB, mask = self.load_from_df(idx)

        if np.random.random() < self.shuffle_ratio and self.phase == "train":
            imgA, imgB = imgB, imgA


        if self.color_trans is not None:
            imgA = self.color_trans(image = imgA)["image"]
            imgB = self.color_trans(image = imgB)["image"]

        if self.trans is not None:
            aug = self.trans(image = imgA, imageB = imgB, mask = mask)
            imgA = aug['image']
            imgB = aug['imageB']
            mask = aug['mask']

        return (imgA, imgB), mask

    def prepare_mosaic_item(self, idx):
        if np.random.rand() < self.mosaic_prob:
            idxes = [idx] + np.random.randint(0, len(self), 3).tolist()
            items = [self.prepare_item(idx) for idx in idxes]
            (imgA, imgB), mask = self.mosaic(items)
        else:
            (imgA, imgB), mask = self.prepare_item(idx)

        if self.post_trans is not None:
            aug = self.post_trans(image = imgA, imageB = imgB, mask = mask)
            imgA = aug['image']
            imgB = aug['imageB']
            mask = aug['mask']

        return (imgA, imgB), mask  

    def mosaic(self, items):
        def random_crop(img, w, h):
            x = int(np.random.randint(0, img.shape[1] - w, 1))
            y = int(np.random.randint(0, img.shape[0] - h, 1))
            return x, x + w, y, y + h
        (base_img, base_imgB), base_mask = items[0]
        center_x = int(round(np.random.uniform(*self.mosaic_center) * base_img.shape[1]))
        center_y = int(round(np.random.uniform(*self.mosaic_center) * base_img.shape[0]))
        for i in range(4):
            if i == 0:
                x1, x2, y1, y2 = random_crop(items[i][1], center_x, center_y)
                base_img[:center_y,:center_x] = items[i][0][0][y1:y2, x1:x2]
                base_imgB[:center_y,:center_x] = items[i][0][1][y1:y2, x1:x2]
                base_mask[:center_y,:center_x] = items[i][1][y1:y2, x1:x2]
            elif i == 1:
                x1, x2, y1, y2 = random_crop(items[i][1], base_img.shape[1] - center_x, center_y)
                base_img[:center_y,center_x:] = items[i][0][0][y1:y2, x1:x2]
                base_imgB[:center_y,center_x:] = items[i][0][1][y1:y2, x1:x2]
                base_mask[:center_y,center_x:] = items[i][1][y1:y2, x1:x2]
            elif i == 2:
                x1, x2, y1, y2 = random_crop(items[i][1], center_x, base_img.shape[0] - center_y)
                base_img[center_y:,:center_x] = items[i][0][0][y1:y2, x1:x2]
                base_imgB[center_y:,:center_x] = items[i][0][1][y1:y2, x1:x2]
                base_mask[center_y:,:center_x] = items[i][1][y1:y2, x1:x2]
            elif i == 3:
                x1, x2, y1, y2 = random_crop(items[i][1], base_img.shape[1] - center_x, base_img.shape[0] - center_y)
                base_img[center_y:,center_x:] = items[i][0][0][y1:y2, x1:x2]
                base_imgB[center_y:,center_x:] = items[i][0][1][y1:y2, x1:x2]
                base_mask[center_y:,center_x:] = items[i][1][y1:y2, x1:x2]
        return (base_img, base_imgB), base_mask

    def __getitem__(self, idx):
        if self.use_mosaic:
            (imgA, imgB), mask = self.prepare_mosaic_item(idx)
        else:
            (imgA, imgB), mask = self.prepare_item(idx)

        img = torch.cat([imgA, imgB], dim = 0)

        return img, mask.long()
