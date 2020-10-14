from abc import *

import numpy as np
import torch
import random

data = np.array([])
inited = False

def init(filepath):
    global data
    data = np.load(filepath, allow_pickle=True).tolist()

def get_chunked_generator(train, cfg):
    assert cfg.keypoint in ['detectron', 'cpn', 'gt']
    assert cfg.receptive_field is not None
    assert cfg.receptive_field % 2 == 1

    subjects = {'train': cfg.train_subjects.replace(' ', '').split(','), 'test': cfg.test_subjects.replace(' ', '').split(',')}
    if cfg.padding is True:
        pad = cfg.receptive_field // 2
    elif type(cfg.padding) == int:
        pad = cfg.padding
    elif cfg.padding is None or cfg.padding is False:
        pad = 0
    else:
        raise Exception('Type of padding must be one of boolean, int, or None.')

    sample_slice = lambda i: (
        slice(i - cfg.receptive_field // 2, i + cfg.length + cfg.receptive_field // 2),
        slice(i, i + cfg.length)
    )

    subs = subjects['train' if train else 'test']

    valid_data_array = []
    slices = []
    for subject in subs:
        for action in data[subject].keys():
            pos2d = data[subject][action]['2d'][cfg.keypoint][
                'normalized' if cfg.normalized else 'unnormalized']
            pos3d = data[subject][action]['3d']['gt']
            assert pos2d.shape[:-1] == pos3d.shape[:-1]

            cam_num = pos2d.shape[0]

            for cam in range(cam_num):
                padded2d = insert_pad(
                    pos2d[cam] if cfg.preprocessor is None else cfg.preprocessor(pos2d[cam]), pad)
                padded3d = insert_pad(pos3d[cam], pad)

                slice_offset = sum([arr.shape[0] for arr in valid_data_array])

                # since the length of the last sample is shorter, the frames of the last sample may be weighter more.
                # for the above reason, the last sample is ignored.
                slices += [
                    sample_slice(slice_offset + cfg.receptive_field // 2 + idx * cfg.length)
                    for idx in range((padded2d.shape[0] - cfg.receptive_field + 1) // cfg.length)
                ]
                valid_data_array.append(np.concatenate([padded2d, padded3d], axis=-1))

    valid_data_array = torch.from_numpy(np.concatenate(valid_data_array, axis=0)).float()

    def gen():
        random.shuffle(slices)
        for i in range(len(slices) // cfg.batch_size):
            x_batch = torch.stack([valid_data_array[slc[0], ..., :2] for slc in slices[i * cfg.batch_size:(i + 1) * cfg.batch_size]])
            y_batch = torch.stack([valid_data_array[slc[1], ..., 2:] for slc in slices[i * cfg.batch_size:(i + 1) * cfg.batch_size]])
            yield (x_batch.cuda(), y_batch.cuda()) if cfg.cuda else (x_batch, y_batch)
        x_batch = torch.stack([valid_data_array[slc[0], ..., :2] for slc in slices[(len(slices) // cfg.batch_size) * cfg.batch_size:-1]])
        y_batch = torch.stack([valid_data_array[slc[1], ..., 2:] for slc in slices[(len(slices) // cfg.batch_size) * cfg.batch_size:-1]])
        yield (x_batch.cuda(), y_batch.cuda()) if cfg.cuda else (x_batch, y_batch)

    return gen

def get_unchunked_generator(train=False, shuffle=False):
    assert inited
    assert cfg.keypoint in ['detectron', 'cpn', 'gt']

    sample_slice = lambda i: (
        slice(i - cfg.receptive_field // 2, i + cfg.length + cfg.receptive_field // 2),
        slice(i, i + cfg.length)
    )

    subs = subjects['train' if train else 'test']

    valid_data_array = []
    slices = []
    for subject in subs:
        for action in data[subject].keys():
            pos2d = data[subject][action]['2d'][cfg.keypoint]['normalized' if cfg.normalized else 'unnormalized']
            pos3d = data[subject][action]['3d']['gt']
            assert pos2d.shape[:-1] == pos3d.shape[:-1]

            cam_num = pos2d.shape[0]

            for cam in range(cam_num):
                processed_2d = pos2d[cam] if cfg.preprocessor is None else cfg.preprocessor(pos2d[cam])
                processed_3d = pos3d[cam]
                valid_data_array.append(np.concatenate([processed_2d, processed_3d], axis=-1))

    valid_data_array = torch.from_numpy(np.concatenate(valid_data_array, axis=0)).float()

    def gen():
        if shuffle:
            random.shuffle(slices)
        for i in range(len(slices) // cfg.batch_size):
            batch = valid_data_array[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            x_batch = batch[..., :2]
            y_batch = batch[..., 2:]
            yield (x_batch.cuda(), y_batch.cuda()) if cfg.cuda else (x_batch, y_batch)
        batch = valid_data_array[(len(slices) // cfg.batch_size) * cfg.batch_size:-1]
        x_batch = batch[..., :2]
        y_batch = batch[..., 2:]
        yield (x_batch.cuda(), y_batch.cuda()) if cfg.cuda else (x_batch, y_batch)

    return gen

def insert_pad(x, pad):
    assert type(pad) == int
    return np.concatenate([x[0][np.newaxis, ...]] * pad + [x] + [x[-1][np.newaxis, ...]] * pad, axis=0)