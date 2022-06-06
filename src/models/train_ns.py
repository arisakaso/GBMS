import copy
from functools import partial

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from src.data.data_module_ns import MultiStepDataModule
from src.models.unet import UNetCustom
from src.solvers.jacobi2d import Jacobi2d
from src.solvers.ns_torch import (
    build_up_RHS,
    compute_velocity,
    prepare_x,
    solve_pressure_poisson,
    solve_pressure_poisson_train,
)
from src.solvers.sor2d import SOR2d
from src.utils.utils import initialize_parameters_custom_unet, relative_l2_error
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import wandb

pl.seed_everything(seed=1, workers=True)


class MetaPDESolver2D(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf)
        pl.seed_everything(seed=1, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # META LEARNER SETUP
        self.meta_learner = UNetCustom(
            in_channels=self.hparams.in_channels,
            out_channels=1,
            channels=self.hparams.channels,
            kernel=self.hparams.kernel,
            dropout=self.hparams.dropout,
            input_drop=self.hparams.input_drop,
        )

        # initialize
        pl.seed_everything(seed=1, workers=True)
        if self.hparams.scale:
            initializer = partial(initialize_parameters_custom_unet, scale=self.hparams.scale)  # ok seed
            self.meta_learner.apply(initializer)  # ok seed

        self.loss = relative_l2_error
        self.alpha = self.hparams.alpha

        # SOLVER SETUP
        self.h = 1 / (self.hparams.size - 1)
        self.jacobi = Jacobi2d(h=self.h)  # to use gpu
        self.sor = SOR2d(h=self.h, omega=self.hparams.omega, learn_omega=False)  # to use gpu
        if self.hparams.solver == "jacobi":
            self.solver = self.jacobi
        elif self.hparams.solver == "sor":
            self.solver = self.sor
        else:
            self.solver = None

        # BCs
        ## pressure
        dbc = torch.zeros((1, 1, self.hparams.size, self.hparams.size))
        dbc[:, :, :, -1] = 1  # rigth edge
        nbc = torch.zeros((1, 1, self.hparams.size, self.hparams.size))
        nbc[:, :, 0, :] = 1
        nbc[:, :, -1, :] = 1
        nbc[:, :, :, 0] = 1
        self.dbc = dbc.cuda()
        self.nbc = nbc.cuda()

        # Sim params
        self.rho = 1
        self.nu = 0.01
        self.dt = 0.001

    def on_fit_start(self):
        pl.seed_everything(seed=wandb.config.seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(True)

    def forward(self, x):  # channel = (f, dbc, nbc, pp, pn - pn-1, fn - fn-1)
        f = x[:, [0]]
        dbc = x[:, [1]]
        nbc = x[:, [2]]
        p0 = self.meta_learner(x)
        p_hat = self.solver(f=f, dbc=dbc, nbc=nbc, num_iter=self.hparams.num_iter, u0=p0)
        return p_hat

    def training_step(self, batch, batch_idx):
        fs, ps, _, us, vs, u_ins, v_ins, ys = batch
        loss = 0
        u = us[:, [-1]]
        v = vs[:, [-1]]

        for i in range(self.hparams.train_step):
            if self.hparams.num_iter == "random":
                num_iter = np.random.randint(0, 125)
            else:
                num_iter = self.hparams.num_iter
            f = build_up_RHS(self.rho, self.dt, u, v, self.h, self.h)
            fs = torch.cat((fs, f), axis=1)
            x = prepare_x(fs, self.dbc, self.nbc, ps, p_ress=None, num_iter=num_iter).clone().detach()
            factor = (torch.sqrt(torch.sum(x[:, [0], :, :] ** 2, axis=[2, 3], keepdim=True))) / float(
                self.hparams.normalize_factor
            )
            x = x / factor
            p, _, _, _, _ = solve_pressure_poisson_train(x, num_iter, 1e-15, self, self.solver)
            p = p * factor
            ps = torch.cat((ps, p), axis=1)

            loss_p = self.loss(ys[:, [i]], p)
            self.log(f"train_loss_p{i:02d}", loss_p, on_epoch=True, on_step=False, logger=True)
            loss += loss_p / self.hparams.train_step

            u, v = compute_velocity(u, v, p, self.h, self.h, self.rho, self.nu, self.dt, u_ins[:, i], v_ins[:, i])

        self.log(f"train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        if isinstance(self.solver, SOR2d):
            self.log(f"omega", self.solver.update.omega, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        fs, ps, p_ress, us, vs, u_ins, v_ins, ys = batch  # (b, c, w, h)
        loss = 0
        u = us[:, [-1]]
        v = vs[:, [-1]]
        if self.hparams.num_iter == "random":
            num_iter = 25
        else:
            num_iter = self.hparams.num_iter
        for i in range(self.hparams.train_step):
            f = build_up_RHS(self.rho, self.dt, u, v, self.h, self.h)
            fs = torch.cat((fs, f), axis=1)

            x = prepare_x(fs, self.dbc, self.nbc, ps, p_ress, num_iter)
            factor = (torch.sqrt(torch.sum(x[:, [0], :, :] ** 2, axis=[2, 3], keepdim=True))) / float(
                self.hparams.normalize_factor
            )
            x = x / factor
            p, k, err, total_err, p_res = solve_pressure_poisson_train(x, num_iter, 1e-15, self, self.solver)
            p = p * factor
            ps = torch.cat((ps, p), axis=1)
            loss_p = self.loss(ys[:, [i]], p)
            self.log(f"val_loss_p{i:02d}", loss_p, on_epoch=True, on_step=False, logger=True)
            loss += loss_p / self.hparams.train_step

            u, v = compute_velocity(u, v, p, self.h, self.h, self.rho, self.nu, self.dt, u_ins[:, i], v_ins[:, i])

        self.log(f"val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        meta_learners = {"trained": self}  # , "baseline": None}
        solvers = {"sor": self.sor}
        for kind, meta_learner in meta_learners.items():
            for solver_name, solver in solvers.items():
                for num_iter in [0, 4, 16, 64]:
                    print(kind, solver_name, num_iter)
                    fs, ps, p_ress, us, vs, u_ins, v_ins, ys = copy.copy(batch)  # (b, c, w, h)
                    loss = 0
                    u = us[:, [-1]]
                    v = vs[:, [-1]]
                    for i in range(self.hparams.test_step):
                        f = build_up_RHS(self.rho, self.dt, u, v, self.h, self.h)
                        fs = torch.cat((fs, f), axis=1)

                        x = prepare_x(fs, self.dbc, self.nbc, ps, p_ress, num_iter).clone()
                        factor = (torch.sqrt(torch.sum(x[:, [0], :, :] ** 2, axis=[2, 3], keepdim=True))) / float(
                            self.hparams.normalize_factor
                        )
                        x = x / factor

                        p, k, err, total_err, p_res = solve_pressure_poisson(x, num_iter, 1e-10, meta_learner, solver)
                        p = p * factor
                        ps = torch.cat((ps, p), axis=1)
                        p_ress = torch.cat((p_ress, p_res), axis=1)
                        loss_p = self.loss(ys[:, [i]], p)
                        self.log(
                            f"test_loss_{kind}_{solver_name}_p{i:02d}_{num_iter:03d}",
                            loss_p,
                            on_epoch=True,
                            on_step=False,
                            logger=True,
                        )
                        loss += loss_p / self.hparams.test_step

                        u, v = compute_velocity(
                            u, v, p, self.h, self.h, self.rho, self.nu, self.dt, u_ins[:, i], v_ins[:, i]
                        )

                    self.log(
                        f"test_loss_{kind}_{solver_name}_{num_iter:03d}",
                        loss,
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
        if self.hparams.scheduler == "step":
            scheduler = StepLR(optimizer, step_size=self.hparams.decay_step, gamma=0.2, verbose=False)
        elif self.hparams.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, factor=self.hparams.lr_factor, patience=self.hparams.lr_patience)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


if __name__ == "__main__":

    ## LOAD CONF
    conf = yaml.safe_load(open("/root/meta-pde-solver/src/models/config_multistep.yml"))
    # if conf["num_iter"] == 0:
    #     conf["solver"] = None
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
    df = pd.DataFrame()
    for i in range(40):
        temp_df = pd.DataFrame()
        temp_df["step"] = range(wandb.config.use_step[0], wandb.config.use_step[1])
        temp_df["episode"] = i
        df = pd.concat([df, temp_df])

    data_module = MultiStepDataModule(  # ok seed
        root_dir=wandb.config.root_dir,
        train_df=df.query(
            f"{wandb.config.train_episode[0]} <= episode <= {wandb.config.train_episode[1]}"
        ).reset_index(drop=True),
        val_df=df.query(f"{wandb.config.val_episode[0]} <= episode <= {wandb.config.val_episode[1]}").reset_index(
            drop=True
        ),
        test_df=df.query(f"{wandb.config.test_episode[0]} <= episode <= {wandb.config.test_episode[1]}").reset_index(
            drop=True
        ),
        train_step=wandb.config.train_step,
        test_step=wandb.config.test_step,
        input_step=wandb.config.input_step,
        batch_size=wandb.config.batch_size,
        num_workers=4,
        dtype=None,
    )

    # MODEL SETUP
    meta_pde_solver = MetaPDESolver2D(dict(wandb.config))

    # TRAINER SETUP
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=wandb.config.es_patience, verbose=True)
    trainer = pl.Trainer(
        max_epochs=wandb.config.max_epochs,
        progress_bar_refresh_rate=1,
        logger=WandbLogger(),
        callbacks=[early_stop_callback],  # prediction_logger,
        gpus=1,
        deterministic=True,
        # fast_dev_run=True,
    )

    # TRAIN
    trainer.fit(model=meta_pde_solver, datamodule=data_module)
    # TEST
    trainer.test()
