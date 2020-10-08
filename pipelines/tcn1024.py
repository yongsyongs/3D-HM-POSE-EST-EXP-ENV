import torch

from common.pipeline import Pipeline
from common.nets import temporal_conv
from common.loss import mpjpe


class pipeline(Pipeline):
    def __init__(self, num_joints, cfg):
        super(pipeline, self).__init__()
        rf = cfg.receptive_field
        fw_cnt = 0
        while rf != 1:
            rf /= 3
            fw_cnt += 1
        self.tcn = temporal_conv.TemporalModel(
            num_joints_in=num_joints, num_joints_out=num_joints, in_features=2, filter_widths=[3] * fw_cnt
        )

        if cfg.cuda:
            self.tcn.to('cuda')

        self.last_loss = 0
        self.last_output = 0

    def forward(self, x, y):
        assert len(x.shape) == 4, f'input shape should be (N T J 2), but got {x.shape}'
        assert x.shape[-1] == 2

        pred = self.tcn(x) # (N T J 3)
        return pred

    def get_loss(self):
        return self.last_loss