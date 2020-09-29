from abc import *

class Pipeline(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x, y):
        pass

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def all_parameters(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)