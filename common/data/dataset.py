from abc import *

import numpy as np
import torch
import random

class Dataset(metaclass=ABCMeta):
    def __init__(self):
        self.num_joints = 0

    def set_num_joints(self, n):
        self.num_joints = n

class H36MDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.subjects = {'train': cfg.train_subjects.replace(' ', '').split(','), 'test': cfg.test_subjects.replace(' ', '').split(',')}
        self.data = np.load(cfg.filepath, allow_pickle=True).tolist()

        self.set_num_joints(self.data['S1']['Eating']['3d']['gt'].shape[-2])

    def get_chunked_generator(self, train=False):
        assert self.cfg.keypoint in ['detectron', 'cpn', 'gt']
        assert self.cfg.receptive_field is not None
        assert self.cfg.receptive_field % 2 == 1

        if self.cfg.padding is True:
            pad = self.cfg.receptive_field // 2
        elif type(self.cfg.padding) == int:
            pad = self.cfg.padding
        elif self.cfg.padding is None or self.cfg.padding is False:
            pad = 0
        else:
            raise Exception('Type of padding must be one of boolean, int, or None.')

        sample_slice = lambda i: (
            slice(i - self.cfg.receptive_field // 2, i + self.cfg.length + self.cfg.receptive_field // 2),
            slice(i, i + self.cfg.length)
        )

        subs = self.subjects['train' if train else 'test']

        valid_data_array = []
        slices = []
        for subject in subs:
            for action in self.data[subject].keys():
                pos2d = self.data[subject][action]['2d'][self.cfg.keypoint][
                    'normalized' if self.cfg.normalized else 'unnormalized']
                pos3d = self.data[subject][action]['3d']['gt']
                assert pos2d.shape[:-1] == pos3d.shape[:-1]

                cam_num = pos2d.shape[0]

                for cam in range(cam_num):
                    padded2d = self.insert_pad(
                        pos2d[cam] if self.cfg.preprocessor is None else self.cfg.preprocessor(pos2d[cam]), pad)
                    padded3d = self.insert_pad(pos3d[cam], pad)

                    slice_offset = sum([arr.shape[0] for arr in valid_data_array])

                    # since the length of the last sample is shorter, the frames of the last sample may be weighter more.
                    # for the above reason, the last sample is ignored.
                    slices += [
                        sample_slice(slice_offset + self.cfg.receptive_field // 2 + idx * self.cfg.length)
                        for idx in range((padded2d.shape[0] - self.cfg.receptive_field + 1) // self.cfg.length)
                    ]
                    valid_data_array.append(np.concatenate([padded2d, padded3d], axis=-1))

        valid_data_array = torch.from_numpy(np.concatenate(valid_data_array, axis=0)).float()

        def gen():
            random.shuffle(slices)
            for i in range(len(slices) // self.cfg.batch_size):
                x_batch = torch.stack([valid_data_array[slc[0], ..., :2] for slc in slices[i * self.cfg.batch_size:(i + 1) * self.cfg.batch_size]])
                y_batch = torch.stack([valid_data_array[slc[1], ..., 2:] for slc in slices[i * self.cfg.batch_size:(i + 1) * self.cfg.batch_size]])
                yield (x_batch.cuda(), y_batch.cuda()) if self.cfg.cuda else (x_batch, y_batch)
            x_batch = torch.stack([valid_data_array[slc[0], ..., :2] for slc in slices[(len(slices) // self.cfg.batch_size) * self.cfg.batch_size:-1]])
            y_batch = torch.stack([valid_data_array[slc[1], ..., 2:] for slc in slices[(len(slices) // self.cfg.batch_size) * self.cfg.batch_size:-1]])
            yield (x_batch.cuda(), y_batch.cuda()) if self.cfg.cuda else (x_batch, y_batch)

        return gen
    def get_unchunked_generator(self, train=False, shuffle=False):
        assert self.cfg.keypoint in ['detectron', 'cpn', 'gt']

        sample_slice = lambda i: (
            slice(i - self.cfg.receptive_field // 2, i + self.cfg.length + self.cfg.receptive_field // 2),
            slice(i, i + self.cfg.length)
        )

        subs = self.subjects['train' if train else 'test']

        valid_data_array = []
        slices = []
        for subject in subs:
            for action in self.data[subject].keys():
                pos2d = self.data[subject][action]['2d'][self.cfg.keypoint]['normalized' if self.cfg.normalized else 'unnormalized']
                pos3d = self.data[subject][action]['3d']['gt']
                assert pos2d.shape[:-1] == pos3d.shape[:-1]

                cam_num = pos2d.shape[0]

                for cam in range(cam_num):
                    processed_2d = pos2d[cam] if self.cfg.preprocessor is None else self.cfg.preprocessor(pos2d[cam])
                    processed_3d = pos3d[cam]
                    valid_data_array.append(np.concatenate([processed_2d, processed_3d], axis=-1))

        valid_data_array = torch.from_numpy(np.concatenate(valid_data_array, axis=0)).float()

        def gen():
            if shuffle:
                random.shuffle(slices)
            for i in range(len(slices) // self.cfg.batch_size):
                batch = valid_data_array[i * self.cfg.batch_size:(i + 1) * self.cfg.batch_size]
                x_batch = batch[..., :2]
                y_batch = batch[..., 2:]
                yield (x_batch.cuda(), y_batch.cuda()) if self.cfg.cuda else (x_batch, y_batch)
            batch = valid_data_array[(len(slices) // self.cfg.batch_size) * self.cfg.batch_size:-1]
            x_batch = batch[..., :2]
            y_batch = batch[..., 2:]
            yield (x_batch.cuda(), y_batch.cuda()) if self.cfg.cuda else (x_batch, y_batch)

        return gen

    def insert_pad(self, x, pad):
        assert type(pad) == int
        return np.concatenate([x[0][np.newaxis, ...]] * pad + [x] + [x[-1][np.newaxis, ...]] * pad, axis=0)