import os
import cv2
import glob
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from .builder import build_trans

class BaseData(Dataset):
    def __init__(self, df, phase, 
            trans = None, 
            organ_order = False, 
            drop_organs = [], 
            resample_query = {},
            balance_key = None,
            **dataset_cfg):
        self.df = df
        self.phase = phase

        self.trans = build_trans(trans.get(self.phase, None))
        if drop_organs:
            self.df = self.df[~self.df.organ.isin(drop_organs)].reset_index(drop = True)

        if organ_order:
            self.label_map = {o: i + 1 for i, o in enumerate(organ_order)}
        else:
            self.label_map = None

        self.df["redirect"] = -1
        self.key_df = []
        self.value_df = []
        for i, query in enumerate(resample_query.get(phase, [])):
            q = query.query
            r = query.ratio
            query_df = self.df.query(q).copy()
            query_df["redirect"] = i

            self.df = self.df[~self.df.index.isin(query_df.index)]

            length = int(round(len(query_df) * r))
            self.key_df.append(query_df.iloc[:length])
            self.value_df.append(query_df.reset_index(drop = True))
        self.df = pd.concat([self.df] + self.key_df).reset_index(drop = True)

        self.balance_key = balance_key if phase == "train" else None

    @staticmethod
    def prepare(**dataset_cfg):
        pass

    def get_labels(self):
        return self.df[self.balance_key]

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        if row["redirect"] != -1:
            idx = np.random.randint(len(self.value_df[row["redirect"]]))
            row = self.value_df[row["redirect"]].loc[idx]

        image_file = row["image_file"]
        mask_file = row["mask_file"]
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)

        if self.label_map:
            mask *= self.label_map[row["organ"]]

        if self.trans is not None:
            aug = self.trans(image = img, mask = mask)
            img = aug['image']
            mask = aug['mask']

        return img, mask.long()


    def __len__(self):
        return self.df.shape[0]