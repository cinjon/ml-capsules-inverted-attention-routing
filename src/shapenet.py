from collections import defaultdict
import math
import os
import random

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

lbl_dict = {
    '02691156': 0, # airplane
    '02747177': 1, #
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
    '04225987': 46, # skateboard
    '04256520': 47,
    '04330267': 48,
    '04379243': 49,
    '04401088': 50,
    '04460130': 51,
    '04468005': 52, # train... actually a lot of these.
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
    """
    NOTE: The Geometric Capsuels paper does a random rotation and translation
    to the viewpoints.
    - Random rotation for the multi-view appointments: [-pi/4, pi/4]
    - Actual input points are translated by random vector in [-1, 1]^3 and
      rotated about a random axis and random angle in -180, 180.
      So ... that's everything. Wtf? It sees every angle. Ok...

    We are doing the translation along stepsize * [1, 1, 1]. When stepsize is
    random in (-1, 1), this is still not a random translation in [-1, 1]^3. To
    get that, we'd need to pick it randomly for *each* direction.
    """
    def __init__(self, path, split='train', npoints=2048, num_frames=3,
                 use_diff_object=True, stepsize_range=None, stepsize_fixed=None,
                 rotation_range=None, rotation_same=None):
        self.path = path
        self.split = split
        self.npoints = npoints # number of points to use
        self.num_frames = num_frames
        self.stepsize_range = stepsize_range
        self.stepsize_fixed = stepsize_fixed
        self.use_diff_object = use_diff_object
        self.rotation_range = rotation_range
        self.rotation_same = rotation_same

        if self.stepsize_fixed is None:
            assert len(self.stepsize_range) == 2

        print('dataset: ', path)
        if 'datasetOriginal' in path:
            self.npy_paths = [os.path.join(self.path, npy)
                              for npy in os.listdir(self.path)]
            if split != 'train':
                # This is because it's a big dataset jsut using train.
                num_subset = int(.1 * len(self.npy_paths))
                self.npy_paths = self.npy_paths[:num_subset]
            self.lbls = None
        else:
            assert self.split in ['train', 'test', 'val']
            self.path = os.path.join(self.path, self.split)
            self.subset_percent = None

            self.npy_paths = []
            self.lbls = []
            folder_names = os.listdir(self.path)
            for folder_name in folder_names:
                npy_names = os.listdir(os.path.join(self.path, folder_name))
                self.npy_paths += [os.path.join(self.path, folder_name, npy_name)
                                   for npy_name in npy_names]
                self.lbls += [lbl_dict[folder_name], ] * len(npy_names)

            if self.split == 'train':
                lbl_count = defaultdict(int)
                for lbl in self.lbls:
                    lbl_count[lbl] += 1
                weights_per_class = {lbl: 1.*len(self.lbls)/count
                                     for lbl, count in lbl_count.items()}
                self.weights_per_index = [weights_per_class[lbl] for lbl in self.lbls]

            self.lbls = np.array(self.lbls)
            self.lbl_dict = {
                i: np.where(self.lbls == i)[0] for i in range(55)
            }
            # 36717 for the 70/20/10 split and 42013 when combining train and val.
            print('Sizes: ', len(self.lbls), len(self.npy_paths))
            if 'dataset5' in path:
                self.subset_count = 5
                self.lbls_used = [8, 9, 28, 31, 41]
            elif 'dataset16' in path:
                self.subset_count = 16
                self.lbls_used = [1, 5, 10, 25, 27, 29, 35, 36, 37, 38, 40, 42, 46, 48, 52, 54]
            else:
                self.subset_count = -1
                self.lbls_used = list(range(55))

    def __getitem__(self, index):
        sample = np.load(self.npy_paths[index])
        sample = self.pc_preprocess(sample)
        sample = np.expand_dims(sample, axis=0).repeat(self.num_frames, 0)

        if self.lbls is not None and len(self.lbls) > 0:
            lbl = self.lbls[index]
            if self.use_diff_object:
                indices = [index]
                while len(indices) < self.num_frames:
                    temp_index = np.random.choice(self.lbl_dict[lbl])
                    if temp_index not in indices:
                        sample[len(indices)] = self.pc_preprocess(np.load(self.npy_paths[temp_index]))
                        indices.append(temp_index)

        # Rotation
        if self.rotation_range is not None:
            # rotation_range is a tuple of (degree_low, degree_high)
            low, high = self.rotation_range
            degrees = np.random.randint(low, high, self.num_frames)
            axes = np.eye(3)[np.random.randint(0, 3, self.num_frames)]
            if self.rotation_same:
                degrees = [degrees[0]] * self.num_frames
                axes = [axes[0]] * self.num_frames
            radians = [d * math.pi / 180. for d in degrees]

            for i in range(self.num_frames):
                rotation = self.rotation_matrix(axes[i], radians[i])
                sample[i] = rotation.apply(sample[i])

        # Translation
        if self.stepsize_fixed:
            stepsize = [self.stepsize_fixed] * self.num_frames
            stepsizes = np.array([[stepsize]*self.num_frames]*3)
        else:
            stepsizes = []
            for i in range(3):
                stepsize = np.random.uniform(
                    self.stepsize_range[0],
                    self.stepsize_range[1],
                    self.num_frames
                )
                stepsizes.append(stepsize)
        stepsizes = np.array(stepsizes).transpose()

        # This is no longer iterative along the translation axis.
        # Doign it that way was an easy way to get the pos/neg.
        for i in range(1, self.num_frames):
            for j in range(3):
                sample[i][j] += stepsizes[i, j]

        # Permute so the channels are in the right dim.
        datum = torch.from_numpy(sample).permute(0, 2, 1).float()

        if self.lbls is not None and len(self.lbls) > 0:
            if self.subset_count > 0:
                lbl = self.lbls_used.index(lbl)
            return datum, lbl
        else:
            return datum, 0

    def pc_preprocess(self, pc):
        # pc = self.total_data[index].copy()
        pc = pc_normalize(pc)

        # Resample
        # NOTE: in the 3d point capsule networks repo replace is True.
        rand_idx = np.random.choice(
            2048, self.npoints, replace=False)
        pc = pc[rand_idx]

        return pc

    @staticmethod
    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        return R.from_rotvec(theta * axis)

    def __len__(self):
        return len(self.npy_paths)
