import numpy as np
import torch

class LinearInterpolation():
    def __init__(self, ratio):
        self.r = ratio

    def __call__(self, x):
        assert type(x) == torch.Tensor or type(x) == np.array

        if self.r == 1:
            return x

        y = torch.zeros_like(x) if type(x) == torch.Tensor else np.zeros_like(x)
        y[0] = x[0]
        for i in range(x.shape[0] - 1):
            y[i + 1] = self.r * x[i + 1] + (1 - self.r) * y[i]

        return y