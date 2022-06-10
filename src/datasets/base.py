from torch.utils.data import Dataset


class BaseData(Dataset):
    def __init__(self, df, phase, **dataset_cfg):
        pass

    @staticmethod
    def prepare(**dataset_cfg):
        """
        prepare dataframes
        """
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass