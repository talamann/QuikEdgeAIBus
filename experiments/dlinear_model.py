import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_windows(y: np.ndarray, input_size: int, h: int):
    n = len(y)
    X = np.empty((n - input_size - h + 1, input_size), dtype=np.float32)
    Y = np.empty((n - input_size - h + 1, h), dtype=np.float32)
    for i in range(X.shape[0]):
        X[i] = y[i : i + input_size]
        Y[i] = y[i + input_size : i + input_size + h]
    return X, Y


class MovingAverageBlock(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, self.padding), mode="reflect")
        return F.avg_pool1d(x, kernel_size=self.kernel_size, stride=1)


class DLinear(nn.Module):
    def __init__(self, input_size: int = 48, h: int = 6, moving_avg: int = 25):
        super().__init__()
        self.input_size = input_size
        self.h = h
        self.trend_block = MovingAverageBlock(moving_avg)
        self.linear_trend = nn.Linear(input_size, h)
        self.linear_seasonal = nn.Linear(input_size, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trend = self.trend_block(x.unsqueeze(1)).squeeze(1)
        x_seasonal = x - x_trend
        return self.linear_trend(x_trend) + self.linear_seasonal(x_seasonal)