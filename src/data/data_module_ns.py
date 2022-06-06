#%%
import os
import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch._C import dtype
from torch.utils.data import DataLoader, Dataset


#%%
class PressurePoisson2DDataModule(pl.LightningDataModule):
    def __init__(self, train_paths: str, val_paths: str, test_paths, batch_size, tensor_type=None):
        super().__init__()
        self.batch_size = batch_size

        dss = []
        for paths in [train_paths, val_paths, test_paths]:
            ds = []
            for path in paths:
                temp = pickle.load(open(path, "rb"))
                if tensor_type:
                    temp = [(x.type(tensor_type), y.type(tensor_type)) for x, y in temp]

                ds += temp

            dss.append(ds)

        self.train_dataset = dss[0]
        self.valid_dataset = dss[1]
        self.test_dataset = dss[2]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=0, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0, pin_memory=False)


#%%


class MultiStepDataset(Dataset):
    def __init__(self, root_dir, meta_df, train_step, input_step, transform=None, double=False) -> None:
        self.root_dir = root_dir
        self.meta_df = meta_df
        self.train_step = train_step
        self.input_step = input_step
        self.double = double

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        episode = self.meta_df.episode.iloc[idx]
        start_step = self.meta_df.step.iloc[idx]

        # previous steps
        fs = torch.tensor(())
        ps = torch.tensor(())
        p_ress = torch.tensor(())
        us = torch.tensor(())
        vs = torch.tensor(())
        for step in range(start_step - self.input_step, start_step):
            f, p, u, v, _, _ = pickle.load(open(os.path.join(self.root_dir, f"{episode}_{step}.pkl"), "rb"))
            fs = torch.cat([fs, f], axis=1)
            ps = torch.cat([ps, p], axis=1)
            p_ress = torch.cat([p_ress, torch.ones_like(p) * 1e-9], axis=1)  # TODO: generate p_res data
            us = torch.cat([us, u], axis=1)
            vs = torch.cat([vs, v], axis=1)

        # current & future steps
        u_ins = torch.tensor(())
        v_ins = torch.tensor(())
        ys = torch.tensor(())
        for step in range(start_step, start_step + self.train_step):
            f, p, u, v, u_in, v_in = pickle.load(open(os.path.join(self.root_dir, f"{episode}_{step}.pkl"), "rb"))
            u_ins = torch.cat([u_ins, u_in], axis=0)  # labels
            v_ins = torch.cat([v_ins, v_in], axis=0)  # labels
            ys = torch.cat([ys, p], axis=1)  # labels

        fs = torch.squeeze(fs, 0)
        ps = torch.squeeze(ps, 0)
        p_ress = torch.squeeze(p_ress, 0)
        us = torch.squeeze(us, 0)
        vs = torch.squeeze(vs, 0)
        ys = torch.squeeze(ys, 0)
        if not self.double:
            fs = fs.float()
            ps = ps.float()
            p_ress = p_ress.float()
            us = us.float()
            vs = vs.float()
            ys = ys.float()

        return (fs, ps, p_ress, us, vs, u_ins, v_ins, ys)


class MultiStepDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        train_df,
        val_df,
        test_df,
        train_step,
        test_step,
        input_step,
        batch_size,
        num_workers=0,
        dtype=None,
    ):
        super().__init__()
        self.train_ds = MultiStepDataset(root_dir, train_df, train_step, input_step)
        self.val_ds = MultiStepDataset(root_dir, val_df, train_step, input_step)
        self.test_ds = MultiStepDataset(root_dir, test_df, test_step, input_step)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size * 16, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size * 8, num_workers=self.num_workers)


# %%
# df = pd.DataFrame()
# for i in range(3):
#     temp_df = pd.DataFrame()
#     temp_df["step"] = range(1000)
#     temp_df["episode"] = i
#     df = pd.concat([df, temp_df])

# df = df.query("100 <= step < 900")

# ds = MultiStepDataset(root_dir="/root/meta-pde-solver/data_share/raw/test", meta_df=df, train_step=2, input_step=2)
# dl = DataLoader(ds, batch_size=10)
# # %%
# for batch in dl:
#     batch
#     break
# # %%
# r
