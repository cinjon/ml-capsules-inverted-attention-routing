import os
import random
import numpy as np
from glob import glob
from scipy.io.matlab import loadmat

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, makedir_exist_ok


class MovingMNist(Dataset):
    def __init__(self, root, train=True, sequence=False):
        self.root = root
        self.train = train
        self.sequence = sequence

        self._transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1397,), (0.3081,))])

        # Get file name
        if self.train:
            img_filename = "moving_mnist_img_train.npy"
            lbl_filename = "moving_mnist_lbl_train.npy"
        else:
            img_filename = "moving_mnist_img_val.npy"
            lbl_filename = "moving_mnist_lbl_val.npy"

        # Load data
        self.img_data = np.load(
            os.path.join(self.root, img_filename)).transpose((1, 0, 2, 3, 4))
        self.lbl_data = np.load(
            os.path.join(self.root, lbl_filename)).astype(np.float32)

        self.seq_len = self.img_data.shape[1]
        self.img_size = self.img_data.shape[3]

    def __getitem__(self, index):
        # Return a seuqnece
        if self.sequence:
            # Get random frame
            frame_num = random.choice(range(self.seq_len-1))

            # Process frames
            this_frame = self._transforms(self.img_data[index, frame_num, 0])
            next_frame = self._transforms(self.img_data[index, frame_num+1, 0])

            return this_frame, next_frame

        # Return the first frame in a sequence
        else:
            img = self._transforms(self.img_data[index, 0, 0])
            lbl = self.lbl_data[index]
            return img, lbl

    def __len__(self):
        return len(self.img_data)

def MovingMNist_sequence_collate(batch):
    frame1 = torch.stack([item[0] for item in batch if item[2]])
    frame2 = torch.stack([item[1] for item in batch if item[2]])
    return [frame1, frame2]


class affNIST(Dataset):
    def __init__(self, root, train):
        self.root = root
        self.train = train
        if self.train:
            self.paths = glob(os.path.join(root, "training_batches", "1.mat"))
        else:
            self.paths = glob(os.path.join(root, "validation_batches", "1.mat"))
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(12),
            transforms.ToTensor(),
            transforms.Normalize((0.1397,), (0.3081,))])

        self.imgs = []
        self.lbls = []
        for path in self.paths:
            data_arr = loadmat(path)["affNISTdata"][0, 0]
            self.lbls.append(data_arr[5].flatten())
            self.imgs.append(
                data_arr[2].transpose((1, 0)).reshape((-1, 40, 40)))

        self.imgs = np.concatenate(self.imgs)
        self.lbls = np.concatenate(self.lbls)

    def __getitem__(self, index):
        return self._transform(self.imgs[index]), self.lbls[index]

    def __len__(self):
        return len(self.lbls)
