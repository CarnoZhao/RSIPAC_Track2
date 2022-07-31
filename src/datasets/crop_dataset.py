import os
import cv2
import glob
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from .builder import build_trans
from .base_dataset import BaseData

class CropData(BaseData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def prepare(image_size = 512, 
            data_dir = "./data/train", 
            **dataset_cfg):
        """
        prepare dataframes
        """
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        image_files = []
        label_files = []
        for d in data_dir:
            image_files += sorted(glob.glob(f"{d}/{image_size}_images/*"))
            label_files += sorted(glob.glob(f"{d}/{image_size}_labels/*"))
        df = pd.DataFrame({"image_file": image_files, "mask_file": label_files})
        df["fold"] = -1
        df["id"] = df.image_file.apply(lambda x: "_".join(os.path.basename(x).split("_")[:2]))
        df["organ"] = df.image_file.apply(lambda x: os.path.basename(x).split("_")[0])
        for fold in range(5):
            split = pd.read_csv(f"./data/train/splits/holdout_{fold}.txt", header = None)
            df.loc[df.id.isin(split.iloc[:,0]), "fold"] = fold
        return df