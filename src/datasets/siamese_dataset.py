import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from .base_dataset import BaseData
from .builder import build_trans
from .concat_dataset import ConcatData

class SiameseData(ConcatData):
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

        return (imgA, imgB), mask.long()
