import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------- Custom Loss Function --------
class ZeroAwareL1Loss(nn.Module):
    def __init__(self, penalty_weight=2.7):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.base_loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        base = self.base_loss(y_pred, y_true)
        zero_mask = torch.isclose(y_true, torch.tensor(0.0, device=y_true.device), atol=1e-3)
        zero_penalty = ((y_pred[zero_mask])**2).mean() if zero_mask.any() else 0.0
        return base + self.penalty_weight * zero_penalty
    

# -------- Residual Block --------

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))

# --------- Model Architecture's --------


# ---------- SAMSResNetCrO Model --------
'''
    Model used for the training of CrO samples. 
    Works pretty well with about 600 ish samples.
'''
class SAMSNetCrO(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# ---------- SAMSResNetSmall Model --------
'''
    Was built to anlyse the small subset of samples of the large dataset.
    For values of SAMA close to zero where the distribution looks uniform.
'''
class SAMSNetSmall(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# ---------- SAMSResNetLarge Model --------
'''
 This is a large model inteded for the training of over
 20k samples.

'''
class SAMSNetLarge(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(128)
        self.final_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.final_layers(x)