import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets.utils import download_url, makedir_exist_ok


class MovingMNist(Dataset):
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


def MovingMNist_collate(batch):
    frame1 = torch.stack([item[0] for item in batch if item[3]])
    frame2 = torch.stack([item[1] for item in batch if item[3]])
    frame3 = torch.stack([item[2] for item in batch if item[3]])
    return [frame1, frame2, frame3]
