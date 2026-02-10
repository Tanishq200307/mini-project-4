# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    MLP for Fashion-MNIST (28x28 -> 784).
    Uses explicit layers to satisfy "nn.Module (not only Sequential)" requirement.
    """
    def __init__(self, hidden_sizes=(256, 128), dropout=0.2, use_batchnorm=True, activation="relu"):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.activation = activation.lower()

        in_dim = 28 * 28
        h1, h2 = hidden_sizes[0], hidden_sizes[1] if len(hidden_sizes) > 1 else None

        self.fc1 = nn.Linear(in_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1) if use_batchnorm else None
        self.drop1 = nn.Dropout(dropout)

        if h2 is not None:
            self.fc2 = nn.Linear(h1, h2)
            self.bn2 = nn.BatchNorm1d(h2) if use_batchnorm else None
            self.drop2 = nn.Dropout(dropout)
            self.out = nn.Linear(h2, 10)
        else:
            self.fc2 = None
            self.bn2 = None
            self.drop2 = None
            self.out = nn.Linear(h1, 10)

    def _act(self, x):
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "leakyrelu":
            return F.leaky_relu(x, negative_slope=0.01)
        if self.activation == "elu":
            return F.elu(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = x.view(x.size(0), -1)  # flatten to (B, 784)

        x = self.fc1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self._act(x)
        x = self.drop1(x)

        if self.fc2 is not None:
            x = self.fc2(x)
            if self.bn2 is not None:
                x = self.bn2(x)
            x = self._act(x)
            x = self.drop2(x)

        logits = self.out(x)
        return logits


class SimpleCNN(nn.Module):
    """
    CNN usually reaches >88-91% on Fashion-MNIST quickly.
    """
    def __init__(self, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.drop = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.bn3 = nn.BatchNorm2d(128)

        # after pool again: 14x14 -> 7x7
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc_out = nn.Linear(256, 10)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(x)

        x = F.relu(self.bn2(self.conv2(x)))  # keep 14x14
        x = self.pool(x)  # 14x14 -> 7x7
        x = self.drop(x)

        x = F.relu(self.bn3(self.conv3(x)))  # 7x7 stays 7x7
        x = self.drop(x)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop(x)
        logits = self.fc_out(x)
        return logits
