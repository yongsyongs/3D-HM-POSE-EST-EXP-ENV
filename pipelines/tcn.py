import torch

from common.pipeline import Pipeline
from common.nets import temporal_conv
from common.loss import mpjpe


class pipeline(Pipeline):
    def __init__(self, cfg):
        super(pipeline, self).__init__()
        self.tcn = temporal_conv.TemporalModel(**cfg.pipeline['tcn_parameters'])

        if cfg.cuda:
            self.tcn.to('cuda')

        self.last_loss = 0
        self.last_output = 0

    def forward(self, x, y):
        assert len(x.shape) == 4, f'input shape should be (N T J 2), but got {x.shape}'
        assert x.shape[-1] == 2

        pred = self.tcn(x) # (N T J 3)
        self.last_loss = mpjpe(pred, y)
        self.last_output = pred
        return pred

    def get_loss(self):
        return self.last_loss