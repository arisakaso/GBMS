from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from src.data.data_module import Poisson1DDataModule
from src.models.unet import UNet1D
from src.solvers.jacobi1d import Jacobi1d
from src.solvers.sor1d import SOR1d
from src.utils.utils import initialize_parameters_unet1d, relative_l2_error
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

import wandb


class GBMS1D(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf)

        # META LEARNER SETUP
        self.meta_learner = UNet1D(
            in_channels=self.hparams.in_channels,
            out_channels=1,
            channels=self.hparams.channels,
            dropout=self.hparams.dropout,
            kernel=self.hparams.kernel,
            input_drop=self.hparams.input_drop,
        )

        # initialize
        if self.hparams.scale:
            initializer = partial(initialize_parameters_unet1d, scale=float(self.hparams.scale))  # ok seed
            self.meta_learner.apply(initializer)

        # SOLVER SETUP
        self.h = 1 / (self.hparams.size - 1)
        self.jacobi = Jacobi1d(h=self.h)  # to use gpu
        self.sor = SOR1d(h=self.h, omega=self.hparams.omega, learn_omega=False)  # to use gpu
        if self.hparams.solver == "jacobi":
            self.solver = self.jacobi
        elif self.hparams.solver == "sor":
            self.solver = self.sor

    def on_fit_start(self):
        pl.seed_everything(seed=wandb.config.seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(True)

    def forward(self, x):
        factor = (torch.sqrt(torch.sum(x[:, [0], :] ** 2, axis=2, keepdim=True))) / float(
            self.hparams.normalize_factor
        )
        x = x / factor
        f = x[:, [0]].clone()
        dbc = x[:, [1]].clone()
        u0 = self.meta_learner(x)
        p_hat = self.solver(f=f, dbc=dbc, num_iter=self.hparams.num_iter, u0=u0)
        return p_hat * factor

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        rl2_loss = relative_l2_error(y, y_hat)
        mse_loss = F.mse_loss(y, y_hat)
        self.log(f"train_rl2", rl2_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"train_mse", mse_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        if self.hparams.loss == "mse":
            return mse_loss
        elif self.hparams.loss == "rl2":
            return rl2_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        rl2_loss = relative_l2_error(y, y_hat)
        mse_loss = F.mse_loss(y, y_hat)
        self.log(f"val_rl2", rl2_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"val_mse", mse_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"val_mse", mse_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        # self.log(f"val_loss", rl2_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        factor = (torch.sqrt(torch.sum(x[:, [0], :] ** 2, axis=2, keepdim=True))) / float(
            self.hparams.normalize_factor
        )
        x = x / factor
        f = x[:, [0]].clone()
        dbc = x[:, [1]].clone()
        u0 = self.meta_learner(x)
        for solver in ["jacobi", "sor"]:
            for nit in [0, 4, 16, 64, 256]:
                if solver == "jacobi":
                    y_hat = self.jacobi(f=f.clone(), dbc=dbc.clone(), num_iter=nit, u0=u0.clone()) * factor
                elif solver == "sor":
                    y_hat = self.sor(f=f.clone(), dbc=dbc.clone(), num_iter=nit, u0=u0.clone()) * factor
                rl2_loss = relative_l2_error(y, y_hat)
                mse_loss = F.mse_loss(y, y_hat)
                self.log(
                    f"test_rl2_{solver}_{nit:03d}", rl2_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
                )
                self.log(
                    f"test_mse_{solver}_{nit:03d}", mse_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
                )
        u0 = dbc.clone()
        for solver in ["jacobi", "sor"]:
            for nit in [0, 4, 16, 64, 256]:
                if solver == "jacobi":
                    y_hat = self.jacobi(f=f, dbc=dbc, num_iter=nit, u0=u0.clone()) * factor
                elif solver == "sor":
                    y_hat = self.sor(f=f, dbc=dbc, num_iter=nit, u0=u0.clone()) * factor
                rl2_loss = relative_l2_error(y, y_hat)
                mse_loss = F.mse_loss(y, y_hat)
                self.log(
                    f"test_rl2_{solver}_baseline_{nit:03d}",
                    rl2_loss,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    f"test_mse_{solver}_baseline_{nit:03d}",
                    mse_loss,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=True,
                    logger=True,
                )

    def configure_optimizers(self):
        if self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = None

        if self.hparams.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.lr_decay_factor,
                patience=self.hparams.lr_patience,
                verbose=True,
            )
        elif self.hparams.scheduler == "step":
            scheduler = StepLR(optimizer, step_size=self.hparams.decay_step, gamma=0.2, verbose=False)
        elif self.hparams.scheduler == "cos":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.t_max, verbose=False)
        elif self.hparams.scheduler == "cos_warm":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.t_0)
        elif self.hparams.scheduler == "onecycle":
            scheduler = OneCycleLR(optimizer, max_lr=self.hparams.lr * 10, total_steps=300)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_mse",
        }


if __name__ == "__main__":

    ## LOAD CONF
    conf = yaml.safe_load(open("/root/meta-pde-solver/src/models/config_poisson1d.yml"))
    wandb.init(config=conf, project=conf["project"])

    # FIX SEED
    pl.seed_everything(seed=wandb.config.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(True)

    if wandb.config.double:
        tensor_type = torch.float64
        torch.set_default_dtype(tensor_type)
    else:
        tensor_type = torch.float32
        torch.set_default_dtype(tensor_type)

    # DATA SETUP

    data_module = Poisson1DDataModule(
        data_path=wandb.config.data_path,
        num_data=wandb.config.num_data,
        batch_size=wandb.config.batch_size,
        normalize=wandb.config.normalize,
    )

    # MODEL SETUP
    meta_pde_solver = GBMS1D(dict(wandb.config))

    # TRAINER SETUP
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop_callback = EarlyStopping(monitor="val_mse", patience=wandb.config.es_patience, verbose=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_mse", verbose=True, save_last=True)
    trainer = pl.Trainer(
        max_epochs=wandb.config.max_epochs,
        progress_bar_refresh_rate=1,
        logger=WandbLogger(),
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],  # prediction_logger,
        gpus=1,
        deterministic=True,
        # fast_dev_run=True,
    )

    # TRAIN
    trainer.fit(model=meta_pde_solver, datamodule=data_module)
    # TEST
    # trainer.test(ckpt_path=checkpoint_callback.dirpath + "/last.ckpt")
    trainer.test()
