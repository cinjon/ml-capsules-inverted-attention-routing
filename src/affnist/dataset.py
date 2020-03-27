from glob import glob
import os
import numpy as np

from scipy.io.matlab import loadmat
from torch.utils.data import Dataset


class AffNist(Dataset):
    def __init__(self, root, train, subset=False, transform=None):
        self.root = root
        self.train = train
        self.subset = subset
        self.transform = transform
        if self.train:
            self.paths = glob(os.path.join(root, "training_batches", "*.mat"))
        else:
            self.paths = glob(os.path.join(root, "validation_batches", "*.mat"))

        if self.subset:
            self.paths = self.paths[:1]

        self.imgs = []
        self.lbls = []
        for path in self.paths:
            data_arr = loadmat(path)["affNISTdata"][0, 0]
            self.lbls.append(data_arr[5].flatten())
            self.imgs.append(
                data_arr[2].transpose((1, 0)).reshape((-1, 40, 40)))

        self.imgs = np.concatenate(self.imgs)
        self.lbls = np.concatenate(self.lbls)

        self.indices_list = []
        for i in range(10):
            self.indices_list.append(
                np.argwhere(self.lbls == i).flatten())

    def __getitem__(self, index):
        img = self.imgs[index]
        lbl = self.lbls[index]
        if self.transform:
            img = self.transform(img)
        return img, lbl

    def __len__(self):
        return len(self.lbls)
