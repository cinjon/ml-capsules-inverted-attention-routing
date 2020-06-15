import os
import h5py
import numpy as np

import torch

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = torch.from_numpy(f['data'][:]).permute(0, 2, 1)
    label = torch.from_numpy(f['label'][:])
    return (data, label)

class ModelNet(torch.utils.data.Dataset):
    def __init__(self, root, npoints=2048, train=True):
        self.root=root
        if train:
            self.list_filename = os.path.join(self.root, 'train_files.txt')
        else:
            self.list_filename = os.path.join(self.root, 'test_files.txt')
        self.npoints = npoints
        self.h5_files = [line.rstrip() for line in open(self.list_filename)]

        self.data = []
        self.label = []
        for h5_file in self.h5_files:
            temp_data, temp_label = load_h5(os.path.join(self.root, h5_file))
            temp_label = np.squeeze(temp_label)
            self.data.append(temp_data)
            self.label.append(temp_label)

        self.data = torch.cat(self.data)
        self.label = torch.cat(self.label)

    def __getitem__(self, index):
        pcd = self.data[index].unsqueeze(0)
        lbl = self.label[index].long()
        return pcd, lbl

    def __len__(self):
        return len(self.data)
