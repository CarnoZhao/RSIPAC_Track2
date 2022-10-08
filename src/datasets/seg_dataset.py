import os
import cv2
import glob
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from .builder import build_trans

class SegData(Dataset):
    def __init__(self, df, phase, 
            trans = None, 
            resample_query = {},
            balance_key = None,
            **dataset_cfg):
        self.df = df
        self.phase = phase

        self.prepare_trans(trans, phase)
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

    def prepare_trans(self, trans, phase):
        if trans.get(phase, None) is not None and any([_["type"] == "Mosaic" for _ in trans[phase]]):
            self.use_mosaic = True
            mosaic_at = [_['type'] == "Mosaic" for _ in trans[phase]].index(True)
            mosaic = trans[phase][mosaic_at]
            self.mosaic_prob = mosaic.get("p", 0.5)
            self.mosaic_center = mosaic.get("center", (0.25, 0.75))
            pre_trans = trans.get(phase, None)[:mosaic_at]
            post_trans = trans.get(phase, None)[mosaic_at + 1:]
            self.trans = build_trans(
                pre_trans, 
                T = trans.get("type", "A"),
                trans_args = trans.get("trans_args", {}))
            self.post_trans = build_trans(
                post_trans, 
                T = trans.get("type", "A"),
                trans_args = trans.get("trans_args", {}))
        else:
            self.use_mosaic = False
            self.trans = build_trans(
                trans.get(phase, None), 
                T = trans.get("type", "A"),
                trans_args = trans.get("trans_args", {}))

    @staticmethod
    def prepare(**dataset_cfg):
        pass

    def get_labels(self):
        return self.df[self.balance_key]

    def prepare_item(self, idx):
        row = self.df.loc[idx]
        if row["redirect"] != -1:
            idx = np.random.randint(len(self.value_df[row["redirect"]]))
            row = self.value_df[row["redirect"]].loc[idx]

        image_file = row["image_file"]
        mask_file = row["mask_file"]
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)

        if self.trans is not None:
            aug = self.trans(image = img, mask = mask)
            img = aug['image']
            mask = aug['mask']

        return img, mask.long()

    def prepare_mosaic_item(self, idx):
        if np.random.rand() < self.mosaic_prob:
            idxes = [idx] + np.random.randint(0, len(self), 3).tolist()
            items = [self.prepare_item(idx) for idx in idxes]
            img, mask = self.mosaic(items)
        else:
            img, mask = self.prepare_item(idx)

        if self.post_trans is not None:
            aug = self.post_trans(image = img, mask = mask)
            img = aug['image']
            mask = aug['mask']

        return img, mask    

    def mosaic(self, items):
        def random_crop(img, w, h):
            x = int(np.random.randint(0, img.shape[1] - w, 1))
            y = int(np.random.randint(0, img.shape[0] - h, 1))
            return x, x + w, y, y + h
        base_img, base_mask = items[0]
        center_x = int(round(np.random.uniform(*self.mosaic_center) * base_img.shape[1]))
        center_y = int(round(np.random.uniform(*self.mosaic_center) * base_img.shape[0]))
        for i in range(4):
            if i == 0:
                x1, x2, y1, y2 = random_crop(items[i][0], center_x, center_y)
                base_img[:center_y,:center_x] = items[i][0][y1:y2, x1:x2]
                base_mask[:center_y,:center_x] = items[i][1][y1:y2, x1:x2]
            elif i == 1:
                x1, x2, y1, y2 = random_crop(items[i][0], base_img.shape[1] - center_x, center_y)
                base_img[:center_y,center_x:] = items[i][0][y1:y2, x1:x2]
                base_mask[:center_y,center_x:] = items[i][1][y1:y2, x1:x2]
            elif i == 2:
                x1, x2, y1, y2 = random_crop(items[i][0], center_x, base_img.shape[0] - center_y)
                base_img[center_y:,:center_x] = items[i][0][y1:y2, x1:x2]
                base_mask[center_y:,:center_x] = items[i][1][y1:y2, x1:x2]
            elif i == 3:
                x1, x2, y1, y2 = random_crop(items[i][0], base_img.shape[1] - center_x, base_img.shape[0] - center_y)
                base_img[center_y:,center_x:] = items[i][0][y1:y2, x1:x2]
                base_mask[center_y:,center_x:] = items[i][1][y1:y2, x1:x2]
        return base_img, base_mask

    def __getitem__(self, idx):
        if self.use_mosaic:
            img, mask = self.prepare_mosaic_item(idx)
        else:
            img, mask = self.prepare_item(idx)
        return img, mask.long()
        


    def __len__(self):
        return self.df.shape[0]