import torch
import torch.nn as nn
import numpy as np
from src.models.modules import ScaledOutputLayer, TrainableTanh


class MetaLearnerFCN(nn.Module):
    def __init__(self, dropout=0, last_activation="trainalbe_tanh"):
        super(MetaLearnerFCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()
        self.fc1 = nn.Linear(512, 512)

        if last_activation == "identity":
            self.fc2 = nn.Linear(512, 512)
            self.last_activation = nn.Identity()
        elif last_activation == "trainable_tanh":
            self.fc2 = nn.Linear(512, 512)
            self.last_activation = TrainableTanh()
        elif last_activation == "parametric_tanh":
            self.fc2 = ScaledOutputLayer(512)
            self.last_activation = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.last_activation(x)
        return x


class MetaLearnerFCNSin(nn.Module):
    def __init__(self, num_basis):
        super(MetaLearnerFCNSin, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_basis)
        self.activation = nn.Tanh()
        self.basis = torch.tensor(
            [np.sin(np.linspace(0, i * np.pi, 512)) for i in range(num_basis)], dtype=torch.float32
        ).cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x @ self.basis
        return x


class MetaLearnerFCNFourier(nn.Module):
    def __init__(self, num_basis):
        super(MetaLearnerFCNFourier, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_basis)
        self.activation = nn.Tanh()
        sins = [np.sin(np.linspace(0, i * np.pi, 512)) for i in range(num_basis // 2)]
        coss = [np.cos(np.linspace(0, i * np.pi, 512)) for i in range(num_basis // 2)]
        self.basis = torch.tensor(sins + coss, dtype=torch.float32).cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x @ self.basis
        return x


class MetaLearnerConv1DSin(nn.Module):
    def __init__(self, num_basis, dropout=0, num_channels=128):
        super(MetaLearnerConv1DSin, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * num_channels, num_basis)
        self.basis = torch.tensor(
            [np.sin(np.linspace(0, i * np.pi, 512)) for i in range(num_basis)], dtype=torch.float32
        ).cuda()
        self.num_basis = num_basis

    def forward(self, x):
        x = x.reshape(-1, 1, 512)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = x.reshape(-1, self.num_basis) @ self.basis
        return x


class MetaLearnerConv1DFourier(nn.Module):
    def __init__(self, num_basis, dropout=0, num_channels=104, last_activation="identity"):
        super(MetaLearnerConv1DFourier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.activation = nn.Tanh()
        if last_activation == "identity":
            self.last_activation = nn.Identity()
        elif last_activation == "trainable_tanh":
            self.last_activation = TrainableTanh()
        else:
            self.last_activation = None
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * num_channels, num_basis)
        sins = [np.sin(np.linspace(0, i * np.pi, 512)) for i in range(num_basis // 2)]
        coss = [np.cos(np.linspace(0, i * np.pi, 512)) for i in range(num_basis // 2)]
        self.basis = torch.tensor(sins + coss, dtype=torch.float32).cuda()
        self.num_basis = num_basis

    def forward(self, x):
        x = x.reshape(-1, 1, 512)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.last_activation(x)
        x = x.reshape(-1, self.num_basis) @ self.basis
        return x


class MetaLearnerConv1DLearnable(nn.Module):
    def __init__(self, num_basis, dropout=0, num_channels=93, last_activation="identity"):
        super(MetaLearnerConv1DLearnable, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.activation = nn.Tanh()
        if last_activation == "identity":
            self.last_activation = nn.Identity()
        elif last_activation == "trainable_tanh":
            self.last_activation = TrainableTanh()
        else:
            self.last_activation = None
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * num_channels, num_basis)
        self.multiply_basis = nn.Linear(num_basis, 512)

    def forward(self, x):
        x = x.reshape(-1, 1, 512)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.last_activation(x)
        x = self.multiply_basis(x)
        return x
