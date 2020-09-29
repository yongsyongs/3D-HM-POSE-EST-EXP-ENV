import torch

from common.pipeline import Pipeline
from common.nets import temporal_conv, non_local
from common.loss import mpjpe


class pipeline(Pipeline):
    def __init__(self, num_joints, cfg):
        rf = cfg.receptive_field
        fw_cnt = 0
        while rf != 1:
            rf /= 3
            fw_cnt += 1
        self.tcn = temporal_conv.TemporalModel(
            num_joints_in=num_joints, num_joints_out=num_joints, in_features=2, filter_widths=[3] * fw_cnt
        )
        self.nlb = non_local.NONLocalBlock1D(num_joints * 3)

        if cfg.cuda:
            self.tcn.to('cuda')
            self.nlb.to('cuda')

        self.last_loss = 0
        self.last_output = 0

    def forward(self, x, y):
        assert len(x.shape) == 4, 'input shape should be (N T J 2)'
        assert x.shape[-1] == 2

        pred = self.tcn(x) # (N T J 3)
        pred = pred.view(pred.shape[:2] + (-1,)).permute(0, 2, 1) # (N C T)
        pred = self.nlb(pred) #(N C T)
        pred = pred.premute(0, 2, 1).view(pred.shape[0], pred.shape[2], -1, 3) # (N T J 3)

        self.last_loss = mpjpe(pred, y)
        self.last_output = pred
        return pred

    def get_loss(self):
        return self.last_loss

    def all_parameters(self):
        all_params = []
        all_params += self.tcn.parameters()
        all_params += self.nlb.parameters()
        return all_params


    def train(self):
        self.tcn.train()
        self.nlb.train()

    def eval(self):
        self.tcn.eval()
        self.nlb.eval()