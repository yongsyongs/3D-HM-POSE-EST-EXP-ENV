from common.config import get_configs
from common.experiments.train import train

cfgs = get_configs()

train(cfgs[0])