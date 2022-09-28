import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from .base_dataset import BaseData
from .builder import build_trans

class ConcatData(BaseData):
    def __init__(self, shuffle_ratio = 0.0, color_trans = None, **kwargs):
        super().__init__(**kwargs)
        if color_trans is not None:
            self.color_trans = build_trans(color_trans.get(self.phase, None), color_trans.get("type", "A"))
        else:
            self.color_trans = None
        self.shuffle_ratio = shuffle_ratio            

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

    def __getitem__(self, idx):
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

        img = torch.cat([imgA, imgB], dim = 0)
        

        return img, mask.long()
