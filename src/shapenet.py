import random
import numpy as np

import torch

def shuffle_points(batch_data):
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ShapeNet55(torch.utils.data.Dataset):
    def __init__(self, path, npoints=2048, num_frames=3,
                 momentum_range=(-1, 1), use_diff_object=True):
        self.path = path
        self.npoints = npoints # number of points to use
        self.num_frames = num_frames
        self.momentum_range = momentum_range
        self.use_diff_object = use_diff_object
        assert len(self.momentum_range) == 2

        npz = dict(np.load(self.path))
        self.total_data = npz['data']
        self.original_sample_num = self.total_data.shape[1]
        assert self.npoints <= self.original_sample_num

        self.lbls = npz['lbls']
        self.lbl_dict = {
            i: np.where(self.lbls == i)[0] for i in range(55)}

    def __getitem__(self, index):
        sample = self._get_pc(index)

        lbl = self.lbls[index]

        sample = np.expand_dims(
            sample, axis=0).repeat(self.num_frames, 0)
        if self.use_diff_object:
            indices = [index]
            while len(indices) < self.num_frames:
                temp_index = np.random.choice(self.lbl_dict[lbl])
                if temp_index not in indices:
                    sample[len(indices)] = self._get_pc(temp_index)
                    indices.append(temp_index)

        # Translation
        momentum = np.random.uniform(
            self.momentum_range[0],
            self.momentum_range[1],
            self.num_frames)

        for i in range(1, self.num_frames):
            sample[i] += i * momentum

        return torch.from_numpy(sample), lbl

    def _get_pc(self, index):
        pc = self.total_data[index].copy()
        pc = pc_normalize(pc)

        # Resample
        if self.npoints != self.original_sample_num:
            # NOTE: in the 3d point capsule networks repo replace is
            # True for some reasons
            rand_idx = np.random.choice(
                self.original_sample_num, self.npoints, replace=False)
            pc = pc[rand_idx]

        return pc

    def __len__(self):
        return len(self.total_data)
