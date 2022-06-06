#%%
import itertools
import numbers
import os
from functools import partial

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.nn.functional import relu
import yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from src.data.data_module import RoberDataModule, RoberDataModule2
from src.solvers.newton import NewtonSOR2, NewtonSORJit
from src.solvers.rober import compute_F, compute_J
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch import nn

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import wandb


class MetaLearnerFCN(nn.Module):
    def __init__(self, in_size=7, mid_size=64, last_activation="sigmoid", initial_bias=-5, initial_guess=False):
        super(MetaLearnerFCN, self).__init__()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.initial_guess = initial_guess
        self.fc1 = nn.Linear(in_size, mid_size)
        self.fc2 = nn.Linear(mid_size, mid_size)
        self.fc3 = nn.Linear(mid_size, 1)
        self.fc4 = nn.Linear(mid_size, 3)
        self.bn1 = nn.BatchNorm1d(mid_size)
        self.bn2 = nn.BatchNorm1d(mid_size)
        if last_activation == "sigmoid":
            self.last_activation = nn.Sigmoid()
            torch.nn.init.constant_(self.fc3.bias, initial_bias)
        elif last_activation == "tanh":
            self.last_activation = nn.Tanh()
            torch.nn.init.constant_(self.fc3.bias, initial_bias)

    def forward(self, x):
        y_prev = x[:, :3].clone()
        y_prev.requires_grad = True
        x[:, 3:] = torch.log10(x[:, 3:])
        # x = torch.log10(x)
        # x[:, 4:] /= 1e6
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        omega = self.fc3(x)
        omega = self.last_activation(omega)
        if self.initial_guess:
            dy = self.tanh(self.fc4(x))
            # y = self.tanh(self.fc4(x)) * 0.1 + y_prev
            y = torch.exp(torch.log(y_prev) + dy)
            return omega + 1, y
        else:
            return omega + 1, y_prev


# %%


class GBMSRober(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf)
        # META LEARNER
        self.meta_learner = MetaLearnerFCN(
            mid_size=self.hparams.mid_size,
            last_activation=self.hparams.last_activation,
            initial_bias=self.hparams.initial_bias,
            initial_guess=self.hparams.initial_guess,
        )
        # for params in self.meta_learner.parameters():
        #     params.register_hook(lambda grad: print(grad, grad.shape, grad.isnan().sum()))  # for debugging

        # SOLVER
        self.solver = NewtonSORJit(
            tol=self.hparams.tol,
            maxiter=self.hparams.maxiter,
            slope=self.hparams.slope,
            log=self.hparams.log_ind,
            clamp_range=self.hparams.clamp_range,
            grad_clamp=self.hparams.grad_clamp,
            last_res=self.hparams.last_res,
        )

        # for debbuging
        # self.solver.requires_grad = True
        # self.solver.register_full_backward_hook(lambda *args: print(args))  # for debugging

        self.solver = torch.jit.script(self.solver)

    def loss_res(self, y_hat, F):
        error = torch.linalg.norm(F(y_hat), dim=1)
        # error = torch.clamp(torch.nan_to_num(torch.linalg.norm(F(y_hat), dim=1), nan=1e5, posinf=1e5), max=1e5)
        if self.hparams.log_res:
            return (torch.log10(1 + relu(error - self.solver.tol))).mean()
        elif self.hparams.last_res:
            return (torch.gather(self.solver.error_hist, 1, self.solver.nit.long().unsqueeze(-1))).mean()
        else:
            return relu(error - self.solver.tol).mean()

    def on_fit_start(self):
        pl.seed_everything(seed=wandb.config.seed, workers=True)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        self.logger.experiment.define_metric("val_loss_num", summary="min")
        self.logger.experiment.define_metric("val_loss_res", summary="min")
        self.logger.experiment.define_metric("val_loss", summary="min")
        self.logger.experiment.define_metric("val_loss_compare", summary="min")

    def forward(self, x, F, J):
        omega, y = self.meta_learner(x.clone())
        if self.training:
            omega.register_hook(lambda grad: torch.nan_to_num(grad, nan=0))
            y.register_hook(lambda grad: torch.nan_to_num(grad, nan=0))
        # omega.register_hook(lambda x: print(x, x.shape, x.max(), x.mean(), x.isnan().sum()))  # for debugging
        # y_hat = self.solver(x[:, :3].clone(), F, J, omega)
        if self.hparams.omega:
            y_hat = self.solver(x, omega, y)
        else:
            y_hat = self.solver(x, torch.ones_like(omega) * 1.37, y)

        return y_hat

    def training_step(self, batch, batch_idx):
        if self.hparams.schedule_maxiter:
            self.solver.maxiter = self.current_epoch * 10 + 1
        x, y = batch
        F = partial(
            compute_F,
            y_old=x[:, :3].clone(),
            h=x[:, 3].clone(),
            k1=x[:, 4].clone(),
            k2=x[:, 5].clone(),
            k3=x[:, 6].clone(),
        )
        J = partial(compute_J, h=x[:, 3].clone(), k1=x[:, 4].clone(), k2=x[:, 5].clone(), k3=x[:, 6].clone())
        y_hat = self(x, F, J)
        loss_res = self.loss_res(y_hat, F)
        loss_num = self.solver.nit.mean()
        loss = self.hparams.alpha * loss_res + loss_num

        self.log(f"train_loss_res", loss_res, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"train_loss_num", loss_num, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        F = partial(
            compute_F,
            y_old=x[:, :3].clone(),
            h=x[:, 3].clone(),
            k1=x[:, 4].clone(),
            k2=x[:, 5].clone(),
            k3=x[:, 6].clone(),
        )
        J = partial(compute_J, h=x[:, 3].clone(), k1=x[:, 4].clone(), k2=x[:, 5].clone(), k3=x[:, 6].clone())
        y_hat = self(x, F, J)
        loss_res = self.loss_res(y_hat, F)
        loss_num = self.solver.nit.mean()
        loss = self.hparams.alpha * loss_res + loss_num  # fix alpha to compare other runs
        loss_compare = 1e9 * loss_res + loss_num  # fix alpha to compare other runs

        self.log(f"val_loss_res", loss_res, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"val_loss_num", loss_num, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"val_loss_compare", loss_compare, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        omega, _ = self.meta_learner(x.clone())
        self.log(f"val_omega_mean", omega.mean(), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"val_omega_std", omega.std(), on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def on_test_start(self):
        self.test_solvers = [
            torch.jit.script(NewtonSORJit(tol=1e-6, maxiter=100000, slope=self.hparams.slope, last_res=False)),
            torch.jit.script(NewtonSORJit(tol=1e-9, maxiter=100000, slope=self.hparams.slope, last_res=False)),
            torch.jit.script(NewtonSORJit(tol=1e-12, maxiter=100000, slope=self.hparams.slope, last_res=False)),
        ]

    def test_step(self, batch, batch_idx):
        self.solver.maxiter = 100000
        x, y = batch
        F = partial(
            compute_F,
            y_old=x[:, :3].clone(),
            h=x[:, 3].clone(),
            k1=x[:, 4].clone(),
            k2=x[:, 5].clone(),
            k3=x[:, 6].clone(),
        )
        J = partial(compute_J, h=x[:, 3].clone(), k1=x[:, 4].clone(), k2=x[:, 5].clone(), k3=x[:, 6].clone())

        omega, y = self.meta_learner(x.clone())
        self.log(f"test_omega_mean", omega.mean(), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"test_omega_std", omega.std(), on_epoch=True, on_step=False, prog_bar=True, logger=True)

        for i, tol in enumerate([1e-6, 1e-9, 1e-12]):
            solver = self.test_solvers[i]
            if self.hparams.omega:
                y_hat = solver(x.clone(), omega, y)
            else:
                y_hat = solver(x.clone(), torch.ones_like(omega) * 1.37, y)
            loss_res = relu(torch.linalg.norm(F(y_hat), dim=1) - solver.tol).mean()
            loss_num = solver.nit.mean()
            loss = self.hparams.alpha * loss_res + loss_num
            loss_compare = 1e9 * loss_res + loss_num  # fix alpha to compare other runs

            self.log(f"test_loss_res_{tol}", loss_res, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            self.log(f"test_loss_num_{tol}", loss_num, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            self.log(f"test_loss_{tol}", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            self.log(
                f"test_loss_compare_{tol}", loss_compare, on_epoch=True, on_step=False, prog_bar=True, logger=True
            )

            if self.hparams.test_fixed_omega:
                for omega_ in torch.linspace(1.3, 1.4, 11):
                    y_hat_omega = solver(x.clone(), torch.ones_like(omega) * omega_, x[:, :3].clone())
                    loss_res_o = self.loss_res(y_hat_omega, F)
                    loss_num_o = solver.nit.mean()
                    loss_o = loss_res_o + self.hparams.alpha * loss_num_o
                    self.log(
                        f"test_loss_res_{tol}_o{omega_}",
                        loss_res_o,
                        on_epoch=True,
                        on_step=False,
                        prog_bar=True,
                        logger=True,
                    )
                    self.log(
                        f"test_loss_num_{tol}_o{omega_}",
                        loss_num_o,
                        on_epoch=True,
                        on_step=False,
                        prog_bar=True,
                        logger=True,
                    )
                    self.log(
                        f"test_loss_{tol}_o{omega_}", loss_o, on_epoch=True, on_step=False, prog_bar=True, logger=True
                    )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler == "rlrop":
            scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams.lr_patience, verbose=True, factor=0.2)
        elif self.hparams.scheduler == "multi_step":
            scheduler = MultiStepLR(optimizer, milestones=self.hparams.lr_milestone, gamma=0.2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    param.grad = torch.nan_to_num(param.grad, nan=0)
                    print(f"detected inf or nan values in gradients. not updating model parameters")


def run():
    ## LOAD CONF
    conf = yaml.safe_load(open("/root/meta-pde-solver/src/models/config_rober_numiter2.yml"))
    wandb.init(config=conf, project=conf["project"])

    # FIX SEED
    pl.seed_everything(seed=wandb.config.seed, workers=True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if wandb.config.double:
        tensor_type = torch.float64
        torch.set_default_dtype(tensor_type)
    else:
        tensor_type = torch.float32
        torch.set_default_dtype(tensor_type)

    # DATA SETUP
    file_list = os.listdir(wandb.config.data_path)
    meta_df = pd.DataFrame([os.path.splitext(f)[0].split("_") for f in file_list], columns=["i", "s"]).astype(int)

    train_df = meta_df[(wandb.config.train[0] <= meta_df.i) & (meta_df.i < wandb.config.train[1])]
    val_df = meta_df[(wandb.config.valid[0] <= meta_df.i) & (meta_df.i < wandb.config.valid[1])]
    test_df = meta_df[(wandb.config.test[0] <= meta_df.i) & (meta_df.i < wandb.config.test[1])]
    data_module = RoberDataModule2(
        root_dir=wandb.config.data_path,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        double=wandb.config.double,
        batch_size=wandb.config.batchsize,
        num_workers=wandb.config.num_workers,
    )

    # MODEL SETUP
    gbms_rober = GBMSRober(dict(wandb.config))

    # TRAINER SETUP
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=wandb.config.es_patience, verbose=True, check_finite=True
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", verbose=True, save_last=True)

    trainer = pl.Trainer(
        max_epochs=wandb.config.max_epochs,
        logger=WandbLogger(log_model=True),
        gpus=1,
        gradient_clip_val=wandb.config.gradient_clip_val,
        gradient_clip_algorithm=wandb.config.gradient_clip_algorithm,
        callbacks=[early_stop_callback, checkpoint_callback],  # prediction_logger,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # limit_test_batches=0.1,
        # deterministic=True,
        fast_dev_run=wandb.config.fast_dev_run,
        # terminate_on_nan=True,
        # detect_anomaly=True,
        track_grad_norm=2,
    )

    # TRAIN
    trainer.fit(model=gbms_rober, datamodule=data_module)
    # TEST
    if wandb.config.do_test:
        trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    run()
