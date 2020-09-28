from abc import *

class Pipeline(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x, y):
        pass

    @abstractmethod
    def get_loss(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.pred(*args, **kwargs)