import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", dest = "config", default = "config.yaml", type = str, nargs = "+")
parser.add_argument("--gpus", dest = "gpus", default = "0", type = str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.dataset import get_data
from src.model import get_model
from src.loss import get_loss
from src.optimizer import get_optimizer
from src.metric import get_metric
from src.train import get_trainer

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.model = get_model(self.cfg.model)
        self.criterion = get_loss(self.cfg.loss)
        self.metric = get_metric(self.cfg.metric)

        self.save_hyperparameters(self.cfg)

    def prepare_data(self):
        self.data = get_data(self.cfg.data)
        (self.ds_train, self.ds_valid), (self.dl_train, self.dl_valid) = self.data

    def train_dataloader(self):
        return self.dl_train()

    def val_dataloader(self):
        return self.dl_valid()

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer(self, self.cfg.train)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        for k in loss:
            self.log("train_" + k, loss[k])
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        for k in loss:
            self.log("val_" + k, loss[k], prog_bar = True)
        return y, yhat

    def validation_step_end(self, output):
        outputs = self.metric.preprocess(output)
        return outputs

    def validation_epoch_end(self, outputs):
        for k, v in self.metric(outputs).items():
            self.log(k, v, prog_bar = True)

if __name__ == "__main__":
    for cfg in args.config:
        cfg = OmegaConf.load(cfg)
        pl.seed_everything(cfg.get("seed", 0))

        model = Model(cfg)
        trainer = get_trainer(args, cfg)
        trainer.fit(model)
    