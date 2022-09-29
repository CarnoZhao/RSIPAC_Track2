import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from .builder import build_trans
from .concat_dataset import ConcatData

class SiameseData(ConcatData):
    def __getitem__(self, idx):
        if self.use_mosaic:
            (imgA, imgB), mask = self.prepare_mosaic_item(idx)
        else:
            (imgA, imgB), mask = self.prepare_item(idx)

        return (imgA, imgB), mask.long()
