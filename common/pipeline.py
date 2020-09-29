from abc import *
import torch.nn as nn

class Pipeline(nn.Module):
    def forward(self, x, y):
        pass

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def all_parameters(self):
        pass