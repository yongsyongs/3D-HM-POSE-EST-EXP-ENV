import numpy as np
import torch
from common.loss import wrap_torch_axis

class ExponentialMovingAverage():
    def __init__(self, r):
        # this value is usually set to (1 - 2/(N + 1)
        self.r = r

    def __call__(self, x):
        assert type(x) == torch.Tensor or type(x) == np.array

        if self.r == 1:
            return x

        y = torch.zeros_like(x) if type(x) == torch.Tensor else np.zeros_like(x)
        y[0] = x[0]
        for i in range(x.shape[0] - 1):
            y[i + 1] = self.r * x[i + 1] + (1 - self.r) * y[i]

        return y

class MovingAverage():
    def __init__(self, window_size):
        self.ws = window_size

    def __call__(self, x):
        assert type(x) == torch.Tensor or type(x) == np.array

        if self.ws == 0:
            return x

        y, mean_func, sum_func, stack_func = (torch.zeros_like(x), wrap_torch_axis(torch.mean, 0), wrap_torch_axis(torch.sum, 0), wrap_torch_axis(torch.stack, 0)) \
            if type(x) == torch.Tensor else (np.zeros_like(x), np.mean, np.sum, np.stack)
        y[:self.ws] = x[self.ws]
        y[-self.ws:] = x[-self.ws:]

        # implemetaion using list comprehension
        # y[self.ws:-self.ws] = [mean_func(x[i - self.ws:i + self.ws + 1], axis=0) for i in range(self.ws, x.shape[0] - self.ws)]

        # implemetation using array slicing
        y[self.ws:-self.ws] = sum_func(stack_func([x[i:-(2 * self.ws + 1) + i] for i in range(2 * self.ws + 1)]), axis=0) / (2 * self.ws + 1)

        return y