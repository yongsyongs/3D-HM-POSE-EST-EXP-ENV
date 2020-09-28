from collections.abc import Sequence
import numpy as np
import copy
import yaml

from common.nets import temporal_conv, non_local, lstm
from common.signal_processing import *
from common.dataset import H36MDataset

dataset_dict = {
    'h36m': H36MDataset,
    'None': None
}

preprocessor_dict = {
    'EMA': ExponentialMovingAverage,
    'MA': MovingAverage,
    'None': None
}

class Config():
    def __init__(
        self, epochs=0, lr=0, lr_decay=[0], lr_decay_step=[0], amp=False, cuda=False, 
        dataset='', filepath='', train_subjects='', test_subjects='', preprocessor='', 
        preprocessor_parameter=0, keypoint='', chunked=False, normalized=False, 
        batch_size=0, receptive_field=0, padding=False, length=0, yaml=None, pipeline=''
    ):
        self.yaml = yaml

        self.pipeline = pipeline

        # hyper params
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_step = lr_decay_step

        # learning process
        self.amp = amp
        self.cuda = cuda

        # dataset
        self.dataset = dataset_dict[dataset]
        self.filepath = filepath
        self.train_subjects = train_subjects
        self.test_subjects = test_subjects

        # data generator
        self.preprocessor = preprocessor_dict[preprocessor](preprocessor_parameter)
        self.keypoint = keypoint
        self.chunked = chunked
        self.normalized = normalized
        self.batch_size = batch_size
        self.receptive_field = receptive_field
        self.padding = padding
        self.length = length

    @staticmethod
    def from_yaml(yaml_data):
        args = {
            'yaml': yaml_data,
            'pipeline': yaml_data['Pipeline'][0],
            **yaml_data['HyperParameters'],
            **yaml_data['Process'],
            **{k:v for k, v in yaml_data['Dataset'].items() if k not in ['name', 'Generator']},
            **{k:v for k, v in yaml_data['Dataset']['Generator'].items() if k != 'preprocessor'},
            'dataset': yaml_data['Dataset']['name'],
            'preprocessor': yaml_data['Dataset']['Generator']['preprocessor']['name'],
            'preprocessor_parameter': yaml_data['Dataset']['Generator']['preprocessor']['parameter'],
        }
        
        return Config(**args)


def get_configs(path='config.yaml'):
    f = open(path)
    raw_cfg = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    var_keys = raw_cfg['Variable']
    cfgs = [raw_cfg]
    for var_key in var_keys:
        mvd, mvk = get_nested_dict(cfgs[0], var_key)
        mvs = mvd[mvk]
        tmp_cfgs = []
        for cfg in cfgs:
            copied_cfgs = [copy.deepcopy(cfg) for _ in range(len(mvs))]
            for tmp_cfg, mv in zip(copied_cfgs, mvs):
                d, k = get_nested_dict(tmp_cfg, var_key)
                d[k] = mv
            tmp_cfgs += copied_cfgs
        del cfgs
        cfgs = copy.deepcopy(tmp_cfgs)
        del tmp_cfgs

    return [Config.from_yaml(cfg) for cfg in cfgs]


def get_nested_dict(nested_dict, key):
    keys = key.split('/')
    d = nested_dict
    for k in keys[:-1]:
        d = d[k]
    return d, keys[-1]




if __name__ == '__main__':
    get_configs()