import torch
from torchsummary import summary
from common.loss import mpjpe


def train(cfg):
    dataset = cfg.dataset(cfg)
    pipeline = __import__('common.pipelines.' + cfg.pipeline, fromlist=[cfg.pipeline]).pipeline(dataset.num_joints, cfg)
    