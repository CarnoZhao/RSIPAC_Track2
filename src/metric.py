from .metrics import DiceMetric

def get_metric(cfg):
    """
    def metric_preprocess(val_step_output):
        return output

    def metric(outputs):
        return {"val_metic": ...}
    """
    cfg = cfg.copy()
    if isinstance(cfg, str):
        return eval(cfg)()
    return eval(cfg.pop("type"))(**cfg)
