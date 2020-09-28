from collections.abc import Sequence
import numpy as np
import copy

from common.Model import temporal_conv, non_local, lstm
from common.signal_processing import *

class mvargs():
    def __init__(self, args):
        assert isinstance(args, Sequence)
        self.args = args

class Config():
    # Hyper params
    # Generator paramas
    # Model structure params
    # names
    def __init__(self):
        self.epoch = 100
        self.lr = 1e-3
        self.lr_decay = [0.95]
        self.lr_decay_step = [1]

        self.use_amp = True
        self.cuda = True

        self.preprocessor_dict = {
            'EMA': ExponentialMovingAverage,
            'MA': MovingAverage,
        }
        self.module_dict = {
            'TCN': temporal_conv.TemporalModel,
            'NLB': non_local.NONLocalBlock1D,
            'RNN': lstm.BiDirectionalLSTM
        }

    def set_cuda(self, cuda):
        self.cuda = cuda

    def set_amp_mode(self, amp):
        self.use_amp = amp

    def set_dataset_cfgs(self, dataset, args):
        assert isinstance(args, dict)
        self.dataset = dataset

        args['cuda'] = self.cuda
        args['preprocessor'] = self.preprocessor_dict[args['preprocessor']](**args['preprocessor_args'])
        # batch_size, length, chunked, keypoint, normalized, padding, receptive_field, preprocessor
        self.dataset_args = self._get_args_from_mvarg(args)

    def set_model_cfgs(self, modules, args):
        assert len(modules) == len(args)
        assert isinstance(args, dict)

        _args = copy.deepcopy(args)

        # (1000, 5) => mvlen (2, 5, 50, 2, 1)
        # if mv_exist:
        #     mvargs_len = [len(mvarg.args) if isinstance(mvarg, mvargs) else 1 for mvarg in args]
        #     opt_count = np.prod(mvargs_len)
        #     opts = [[] for _ in range(len(mvargs_len))]
        #     for i, arg_count in enumerate(mvargs_len):
        #         chk_count = np.prod(mvargs_len[:i])
        #         for _ in range(chk_count):
        #             for j in range(arg_count):
        #                 opts[i] += [copy.deepcopy(args[j]) for _ in range(opt_count // chk_count // arg_count)]
        # else:
        #     opts = [[arg] for arg in args]




        modules = [self.module_dict[key] for key in modules]
        module_args = [self._get_args_from_mvarg(arg) for arg in args]
        self.modules = [module(module_arg) for module, module_arg in zip(modules, module_args)]

    # can use config for grid search with this method
    def _get_args_from_mvarg(self, args):
        grid_args = { k: v.args if isinstance(v, mvargs) else [v] for k, v in args.items() }

        opt_count = 1
        for l in [len(v) for v in grid_args.values()]:
            opt_count *= l
        args_array = np.empty((opt_count, len(grid_args.keys())), dtype=np.object)
        a, b = opt_count, 1
        for i, key in enumerate(grid_args.keys()):
            arg_len = len(grid_args[key])
            a = a // arg_len
            args_array[:, i] = np.tile(np.full((a, arg_len), grid_args[key]).transpose().reshape(-1), b)
            b *= arg_len

        return [{k: args_array[i, j] for j, k in enumerate(grid_args.keys())} for i in range(opt_count)]