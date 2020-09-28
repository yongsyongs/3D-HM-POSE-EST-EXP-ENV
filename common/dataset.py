import numpy as np
import torch

import random

all_subjects = 'S1,S5,S6,S7,S8,S9,S11'.split(',')

class H36MDataset():
    def __init__(self, train_subjects='S1,S5,S6,S7,S8', test_subjects='S9,S11',
        filepath='data/h36m_cpn+detectron+gt_2d_normalized+unnormlized_gt_3d.npy'):
        self.subjects = {'train': train_subjects.replace(' ', '').split(','), 'test': test_subjects.replace(' ', '').split(',')}
        self.data = np.load(filepath, allow_pickle=True).tolist()

    def get_generator(
            self, keypoint, normalized=True, batch_size=32, length=1, padding=None,
            chunked=False, receptive_field=None, train=False, cuda=True, preprocessor=None
    ):
        assert keypoint in ['detectron', 'cpn', 'gt']

        if chunked:
            assert receptive_field is not None
            assert receptive_field % 2 == 1
        else:
            receptive_field = 0

        if padding is True:
            pad = receptive_field // 2
        elif type(padding) == int:
            pad = padding
        elif padding is None or padding is False:
            pad = 0
        else:
            raise Exception('Type of padding must be one of boolean, int, or None.')

        as_is = lambda i: slice(i, i + length)
        sample_slice = lambda i: (
            slice(i - receptive_field // 2, i + length + receptive_field // 2) if chunked else as_is(i),
            as_is(i)
        )

        subs = self.subjects['train' if train else 'test']

        valid_data_array = []
        slices = []
        for subject in subs:
            for action in self.data[subject].keys():
                pos2d = self.data[subject][action]['2d'][keypoint]['normalized' if normalized else 'unnormalized']
                pos3d = self.data[subject][action]['3d']['gt']
                assert pos2d.shape[:-1] == pos3d.shape[:-1]

                cam_num = pos2d.shape[0]

                for cam in range(cam_num):
                    padded2d = self.insert_pad(pos2d[cam] if preprocessor is None else preprocessor(pos2d[cam]), pad)
                    padded3d = self.insert_pad(pos3d[cam] if preprocessor is None else preprocessor(pos3d[cam]), pad)

                    slice_offset = sum([arr.shape[0] for arr in valid_data_array])

                    # since the length of the last sample is shorter, the frames of the last sample may be weighter more.
                    # for the above reason, the last sample is ignored.
                    slices += [sample_slice(slice_offset + receptive_field // 2 + idx * length) for idx in range((padded2d.shape[0] - receptive_field + 1) // length)]
                    valid_data_array.append(np.concatenate([padded2d, padded3d], axis=-1))

        valid_data_array = torch.from_numpy(np.concatenate(valid_data_array, axis=0))
        
        def gen():
            random.shuffle(slices)
            for i in range(len(slices) // batch_size):
                x_batch = torch.stack([valid_data_array[slc[0], ..., :2] for slc in slices[i * batch_size:(i + 1) * batch_size]])
                y_batch = torch.stack([valid_data_array[slc[1], ..., 2:] for slc in slices[i * batch_size:(i + 1) * batch_size]])
                yield x_batch.cuda(), y_batch.cuda() if cuda else x_batch, y_batch
            x_batch = torch.stack([valid_data_array[slc[0], ..., :2] for slc in slices[(i + 1) * batch_size:-1]])
            y_batch = torch.stack([valid_data_array[slc[1], ..., 2:] for slc in slices[(i + 1) * batch_size:-1]])
            yield x_batch.cuda(), y_batch.cuda() if cuda else x_batch, y_batch

        return gen

    def insert_pad(self, x, pad):
        assert type(pad) == int
        return np.concatenate([x[0][np.newaxis, ...]] * pad + [x] + [x[-1][np.newaxis, ...]] * pad, axis=0)