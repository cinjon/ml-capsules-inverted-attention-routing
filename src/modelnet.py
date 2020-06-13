import os
import numpy as np

import torch
import torch.nn as nn

label_dic = {
    'airplane': 0,
    'bathtub': 1,
    'bed': 2,
    'bench': 3,
    'bookshelf': 4,
    'bottle': 5,
    'bowl': 6,
    'car': 7,
    'chair': 8,
    'cone': 9,
    'cup': 10,
    'curtain': 11,
    'desk': 12,
    'door': 13,
    'dresser': 14,
    'flower_pot': 15,
    'glass_box': 16,
    'guitar': 17,
    'keyboard': 18,
    'lamp': 19,
    'laptop': 20,
    'mantel': 21,
    'monitor': 22,
    'night_stand': 23,
    'person': 24,
    'piano': 25,
    'plant': 26,
    'radio': 27,
    'range_hood': 28,
    'sink': 29,
    'sofa': 30,
    'stairs': 31,
    'stool': 32,
    'table': 33,
    'tent': 34,
    'toilet': 35,
    'tv_stand': 36,
    'vase': 37,
    'wardrobe': 38,
    'xbox': 39
}

class ModelNet(nn.Module):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train

        folder_names = sorted(os.listdir(self.root))
        split = 'train' if self.train else 'test'

        self.paths = []
        self.lbls = []
        for folder_name in folder_names:
            folder_path = os.path.join(self.root, folder_name, split)
            pcd_names = sorted(os.listdir(folder_path))
            self.paths += [os.path.join(folder_path, pcd_name) for pcd_name in pcd_names]
            self.lbls += [label_dic[folder_name]] * len(pcd_names)

    def __getitem__(self, index):
        print(self.paths[index])
        pcd = torch.from_numpy(np.load(self.paths[index])).permute(1, 0)
        lbl = self.lbls[index]
        return pcd, lbl

    def __len__(self):
        return len(self.paths)
