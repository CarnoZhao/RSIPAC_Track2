import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, StochasticWeightAveraging, LearningRateMonitor
import torch
torch.backends.cudnn.enabled = False
import cv2
cv2.setNumThreads(4)

def get_trainer(args, cfg):

    # logger
    logger = [
        CSVLogger("./logs", 
            name = cfg.name, 
            version = cfg.version, 
            flush_logs_every_n_steps = cfg.train.log_step),
    ]


    # callbacks
    monitor = cfg.train.get("monitor", "valid_metric")
    callbacks = [
        ModelCheckpoint(
            dirpath = os.path.join("./logs", cfg.name, cfg.version),
            filename = '{epoch}_{' + monitor + ':.3f}',
            save_last = True,
            save_top_k = cfg.train.get("save_topk", 3),
            save_weights_only = True,
            mode = "max",
            monitor = monitor),
        RichProgressBar(leave = True),
        LearningRateMonitor('step')
    ]
    if cfg.train.get("swa", False):
        callbacks.append(StochasticWeightAveraging())
        
    # trainer
    grad_clip = cfg.train.get("grad_clip", 0)
    trainer = pl.Trainer(
        accelerator = "gpu",
        gpus = list(range(len(args.gpus.split(",")))), 
        precision = 16, 
        strategy = cfg.train.get("strategy", "dp"),
        sync_batchnorm = cfg.train.get("strategy", "dp") == "ddp",
        gradient_clip_val = grad_clip,
        accumulate_grad_batches = cfg.train.get("grad_acc", 1),
        max_epochs = cfg.train.num_epochs,
        logger = logger,
        callbacks = callbacks,
        log_every_n_steps = cfg.train.log_step,
        check_val_every_n_epoch = cfg.train.val_interval,
    )

    return trainer