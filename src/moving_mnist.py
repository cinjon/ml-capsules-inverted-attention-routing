import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets.utils import download_url, makedir_exist_ok


# For generated data
class MovingMNist(Dataset):
    def __init__(self, root, train=True, sequence=False):
        self.root = root
        self.train = train
        self.sequence = sequence

        self._transforms = Compose([
            ToTensor(),
            Normalize((0.1397,), (0.3081,))])

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


# For test data downloaded from website
class TestMovingMNist(Dataset):
    resources = ("http://www.cs.toronto.edu/~nitish/unsupervised_video/"
                 "mnist_test_seq.npy")
    filename = "mnist_test_seq.npy"

    def __init__(self, root, download=False, seed=0):
        self.root = root
        makedir_exist_ok(self.root)

        if download:
            download_url(
                self.resources, root=self.root)

        self.seed = seed
        random.seed(self.seed)

        self._transforms = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        self.data = np.load(os.path.join(self.root, self.filename))
        self.data = self.data.transpose((1, 0, 2, 3))
        self.seq_len = self.data.shape[1]
        self.image_size = self.data.shape[2]
        self.data = self.data.reshape((-1, self.image_size, self.image_size))

    def __getitem__(self, index):
        # If the chosen frame is at the end of a sequence then ignore it
        if (index+1)%self.seq_len == 0:
            zero_tensor = torch.zeros(1, 64, 64)
            return zero_tensor, zero_tensor, zero_tensor, False

        # Randomly select a frame
        # start = index // self.seq_len * self.seq_len
        # end = start + self.seq_len
        # num = random.choice(
        #     [i for i in range(start, end) if i != index and i != index + 1])
        num = random.choice(
            [i for i in range(len(self.data)) if i != index and i != index + 1])

        # Process frames
        this_frame = self._transforms(self.data[index])
        next_frame = self._transforms(self.data[index+1])
        other_frame = self._transforms(self.data[num])

        return this_frame, next_frame, other_frame, True

    def __len__(self):
        return len(self.data)


def MovingMNist_sequence_collate(batch):
    frame1 = torch.stack([item[0] for item in batch if item[2]])
    frame2 = torch.stack([item[1] for item in batch if item[2]])
    return [frame1, frame2]
