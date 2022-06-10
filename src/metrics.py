import torch
from omegaconf import OmegaConf

def get_metric(cfg):
    """
    def metric_preprocess(val_step_output):
        return output

    def metric(outputs):
        return {"val_metic": ...}
    """

    if isinstance(cfg, str):
        return eval(cfg), eval(cfg + "_preprocess")
    return eval(cfg.type), eval(cfg.type + "_preprocess")
