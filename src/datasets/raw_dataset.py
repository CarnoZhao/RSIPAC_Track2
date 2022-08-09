import os
import cv2
import glob
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from .builder import build_trans
from .base_dataset import BaseData

class RawData(BaseData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def prepare(image_size = 768, 
            data_dir = "./data/train",
            **dataset_cfg):
        """
        prepare dataframes
        """
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        image_files = []
        label_files = []
        data_source = []
        for d in data_dir:
            image_files += sorted(glob.glob(f"{d}/resized_{image_size}_images/*"))
            label_files += sorted(glob.glob(f"{d}/resized_{image_size}_labels/*"))
            data_source += [d] * len(glob.glob(f"{d}/resized_{image_size}_labels/*"))
        df = pd.DataFrame({"image_file": image_files, "mask_file": label_files, "data_dir": data_source})
        df["fold"] = -1
        df["id"] = df.image_file.apply(lambda x: "_".join(os.path.basename(x).split('.')[0].split("_")[:2]))
        df["organ"] = df.image_file.apply(lambda x: os.path.basename(x).split("_")[0])
        for fold in range(5):
            split = pd.read_csv(f"./data/train/splits/holdout_{fold}.txt", header = None)
            df.loc[df.id.isin(split.iloc[:,0]), "fold"] = fold
        return df

    def __getitem__(self, idx):
        image_file = self.df.loc[idx, "image_file"]
        mask_file = self.df.loc[idx, "mask_file"]
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)

        if self.trans is not None:
            aug = self.trans(image = img, mask = mask)
            img = aug['image']
            mask = aug['mask']

        return img, mask.long()


    def __len__(self):
        return self.df.shape[0]
