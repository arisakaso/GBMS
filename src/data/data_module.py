#%%
import os
import pickle
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class Poisson1DDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, num_data, batch_size, double=False, normalize=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_data = num_data

        ds = pickle.load(open(data_path, "rb"))
        ds = [(torch.stack([f, torch.linspace(u[0], u[-1], len(u))]), u.reshape(1, -1)) for f, u in ds]
        # for x, y in ds:
        #     x[1, 1:-2] = 0
        if double:
            pass
        else:
            ds = [(x.type(torch.float32), y.type(torch.float32)) for x, y in ds]
        if normalize:
            ds = [(x / torch.max(x[0]), y / torch.max(x[0])) for x, y in ds]

        self.train_dataset = ds[: self.num_data]
        self.valid_dataset = ds[-20000:-10000]
        self.test_dataset = ds[-10000:]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size * 5, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size * 5, num_workers=0)


class PoissonDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, test_paths, num_data, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.num_data = num_data

        ds = pickle.load(open(data_path, "rb"))
        ds = [(d[0] / 512**2, d[1]) for d in ds]
        self.train_dataset = ds[: self.num_data]
        self.valid_dataset = ds[10000:20000]

        self.test_datasets = []
        self.test_datasets.append(ds[20000:])  # same distribution as train
        for test_path in test_paths:
            test_ds = pickle.load(open(test_path, "rb"))
            test_ds = [(d[0] / 512**2, d[1]) for d in test_ds]
            test_ds = test_ds[-10000:]  # for testing same data
            self.test_datasets.append(test_ds)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=10000, num_workers=0, pin_memory=False)

    def test_dataloader(self):
        return [DataLoader(ds, batch_size=10000, num_workers=0, pin_memory=False) for ds in self.test_datasets]


class Poisson2DDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, test_paths, num_data, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.num_data = num_data

        ds = pickle.load(open(data_path, "rb"))
        # ds = [(d[0].reshape(1, size, size), d[1].reshape(1, size, size)) for d in ds]
        self.train_dataset = ds[: self.num_data]
        self.valid_dataset = ds[10000:20000]

        self.test_datasets = []
        self.test_datasets.append(ds[20000:])  # same distribution as train
        for test_path in test_paths:
            test_ds = pickle.load(open(test_path, "rb"))
            # test_ds = [(d[0].reshape(1, size, size), d[1].reshape(1, size, size)) for d in test_ds]
            test_ds = test_ds[-10000:]  # for testing same data
            self.test_datasets.append(test_ds)

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
        return [
            DataLoader(ds, batch_size=self.batch_size, num_workers=0, pin_memory=False) for ds in self.test_datasets
        ]


class PoissonControlDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        train_df,
        val_df,
        test_df,
        batch_size,
        num_workers=0,
        dtype=None,
    ):
        super().__init__()
        self.train_ds = RoberDataset(root_dir, train_df)
        self.val_ds = RoberDataset(root_dir, val_df)
        self.test_ds = RoberDataset(root_dir, test_df)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            # generator=torch.Generator().manual_seed(42),
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size * 4, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size * 4, num_workers=self.num_workers)


class RoberDataset(Dataset):
    def __init__(self, root_dir, meta_df, double=False, processed=True, transform=None) -> None:
        self.root_dir = root_dir
        self.meta_df = meta_df
        self.double = double
        self.processed = processed

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        g = self.meta_df.g.iloc[idx]
        i = self.meta_df.i.iloc[idx]
        s = self.meta_df.s.iloc[idx]

        if self.processed:
            x, y = pickle.load(open(os.path.join(self.root_dir, f"{g}_{i}_{s}.pkl"), "rb"))
            return x, y

        y, y_old, h, k1, k2, k3, tol = pickle.load(open(os.path.join(self.root_dir, f"{g}_{i}_{s}.pkl"), "rb"))
        x = torch.cat([y_old.squeeze(), torch.Tensor([h, k1, k2, k3])])
        y = y.squeeze()
        if self.double:
            x = x.double()
            y = y.double()
        else:
            x = x.float()
            y = y.float()

        return x, y


class RoberDataset2(Dataset):
    def __init__(self, root_dir, meta_df, double=False, processed=True, transform=None, tol_range=None) -> None:
        self.root_dir = root_dir
        self.meta_df = meta_df
        self.tol_range = tol_range

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        i = self.meta_df.i.iloc[idx]
        s = self.meta_df.s.iloc[idx]

        x, y = pickle.load(open(os.path.join(self.root_dir, f"{i}_{s}.pkl"), "rb"))
        if self.tol_range:
            tol = torch.tensor(np.power(10, np.random.uniform(-5, 1, (num_samples))))
            x = torch.cat([x, torch.Tensor([tol])])

        return x.squeeze(), y.squeeze()


class RoberDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, train_df, val_df, test_df, batch_size, num_workers=0, double=False):
        super().__init__()
        self.train_ds = RoberDataset(root_dir, train_df, double)
        self.val_ds = RoberDataset(root_dir, val_df, double)
        self.test_ds = RoberDataset(root_dir, test_df, double)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)


class RoberDataModule2(pl.LightningDataModule):
    def __init__(self, root_dir, train_df, val_df, test_df, batch_size, num_workers=0, double=False):
        super().__init__()
        self.train_ds = RoberDataset2(root_dir, train_df, double)
        self.val_ds = RoberDataset2(root_dir, val_df, double)
        self.test_ds = RoberDataset2(root_dir, test_df, double)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)


#%%
# import os
# import pickle
# import pandas as pd


# #%%
# data_path = "/root/meta-pde-solver/data/raw/rober/"
# file_list = os.listdir(data_path)
# meta_df = pd.DataFrame([os.path.splitext(f)[0].split("_") for f in file_list], columns=["i", "s"]).astype(int)

# # %%
# train_df = meta_df[meta_df.i < 800]
# val_df = meta_df[(800<=meta_df.i) & (meta_df.i <900)]
# test_df = meta_df[(900<=meta_df.i)]
