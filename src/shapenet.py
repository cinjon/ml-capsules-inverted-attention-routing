import random

import numpy as np
import torch

lbl_dict = {
    '02691156': 0,
    '02747177': 1,
    '02773838': 2,
    '02801938': 3,
    '02808440': 4,
    '02818832': 5,
    '02828884': 6,
    '02843684': 7,
    '02871439': 8,
    '02876657': 9,
    '02880940': 10,
    '02924116': 11,
    '02933112': 12,
    '02942699': 13,
    '02946921': 14,
    '02954340': 15,
    '02958343': 16,
    '02992529': 17,
    '03001627': 18,
    '03046257': 19,
    '03085013': 20,
    '03207941': 21,
    '03211117': 22,
    '03261776': 23,
    '03325088': 24,
    '03337140': 25,
    '03467517': 26,
    '03513137': 27,
    '03593526': 28,
    '03624134': 29,
    '03636649': 30,
    '03642806': 31,
    '03691459': 32,
    '03710193': 33,
    '03759954': 34,
    '03761084': 35,
    '03790512': 36,
    '03797390': 37,
    '03928116': 38,
    '03938244': 39,
    '03948459': 40,
    '03991062': 41,
    '04004475': 42,
    '04074963': 43,
    '04090263': 44,
    '04099429': 45,
    '04225987': 46,
    '04256520': 47,
    '04330267': 48,
    '04379243': 49,
    '04401088': 50,
    '04460130': 51,
    '04468005': 52,
    '04530566': 53,
    '04554684': 54
}


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ShapeNet55(torch.utils.data.Dataset):
    def __init__(self, path, split='train', npoints=2048, num_frames=3,
                 momentum_range=(-1, 1), use_diff_object=True):
        self.path = path
        self.split = split
        self.npoints = npoints # number of points to use
        self.num_frames = num_frames
        self.momentum_range = momentum_range
        self.use_diff_object = use_diff_object
        assert len(self.momentum_range) == 2
        assert self.split in ['train', 'test', 'val']
        self.path = os.path.join(self.path, self.split)

        self.npy_paths = []
        self.lbls = []
        folder_names = os.listdir(self.path)
        for folder_name in folder_names:
            npy_names = os.listdir(os.path.join(self.path, folder_name))
            self.npy_paths += [os.path.join(self.path, folder_name, npy_name) for npy_name in npy_names]
            self.lbls += [lbl_dict[folder_name], ] * len(npy_names)

        self.lbls = np.array(self.lbls)
        self.lbl_dict = {
            i: np.where(self.lbls == i)[0] for i in range(55)}

    def __getitem__(self, index):
        sample = np.load(self.npy_paths[index])
        sample = self.pc_preprocess(sample)

        lbl = self.lbls[index]

        sample = np.expand_dims(
            sample, axis=0).repeat(self.num_frames, 0)
        if self.use_diff_object:
            indices = [index]
            while len(indices) < self.num_frames:
                temp_index = np.random.choice(self.lbl_dict[lbl])
                if temp_index not in indices:
                    sample[len(indices)] = self.pc_preprocess(np.load(self.npy_paths[temp_index]))
                    indices.append(temp_index)

        # Translation
        momentum = np.random.uniform(
            self.momentum_range[0],
            self.momentum_range[1],
            self.num_frames)

        for i in range(1, self.num_frames):
            sample[i] += i * momentum

        return torch.from_numpy(sample), lbl

    def pc_preprocess(self, pc):
        # pc = self.total_data[index].copy()
        pc = pc_normalize(pc)

        # Resample
        # NOTE: in the 3d point capsule networks repo replace is
        # True for some reasons
        rand_idx = np.random.choice(
            2048, self.npoints, replace=False)
        pc = pc[rand_idx]

        return pc

    def __len__(self):
        return len(self.npy_paths)
