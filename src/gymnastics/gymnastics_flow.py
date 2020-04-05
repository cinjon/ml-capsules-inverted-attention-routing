import os
import json
import numpy as np
from glob import glob

import torch

class gymnastics_flow(torch.utils.data.DataLoader):
    def __init__(
        self,
        root,
        bad_list_path,
        nan_list_path,
        magnitude_list_path,
        min_distance=60,
        positive_ratio=0.25,
        transform=None):
        
        self.root = root
        self.bad_list_path = bad_list_path
        self.nan_list_path = nan_list_path
        self.magnitude_list_path = magnitude_list_path
        self.min_distance = min_distance
        self.positive_ratio = positive_ratio
        self.transform = transform
        
        self.get_paths()
        self.get_magnitude_dict()
        self.make_colorwheel()
            
    def get_paths(self):
        bad_flows_set = set()
        
        # Get bad flow list from "still-bad-flows.json"
        with open(self.bad_list_path)as f:
            bad_flows_dict = json.load(f)
        for value in bad_flows_dict.values():
            bad_flows_set.add(value[0][2])
            
        # Get bad flow list from "nan_flow_list.json"
        with open(self.nan_list_path) as f:
            nan_flows_dict = json.load(f)
        for key in nan_flows_dict.keys():
            # bad_flows_set.add(os.path.basename(key[:-1]))
            bad_flows_set.add(key)
            
        # Get npy paths
        self.folder_paths = [
            folder_path for folder_path in sorted(glob(os.path.join(self.root, "*/")))\
                if os.path.basename(folder_path[:-1]) not in bad_flows_set]
        self.data_dict = {}
        for folder_path in self.folder_paths:
            self.data_dict[folder_path] = (sorted(glob(os.path.join(folder_path, "*.npy"))))
        
    def get_magnitude_dict(self):
        self.magnitude_dict = {}
        magnitude = np.load(self.magnitude_list_path, allow_pickle=True)
        for m in magnitude:
            self.magnitude_dict[m[0]] = m[1]
    
    def make_colorwheel(self):
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR
        self.colorwheel = np.zeros((ncols, 3))
        col = 0

        # RY
        self.colorwheel[0:RY, 0] = 255
        self.colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
        col = col+RY
        # YG
        self.colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
        self.colorwheel[col:col+YG, 1] = 255
        col = col+YG
        # GC
        self.colorwheel[col:col+GC, 1] = 255
        self.colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
        col = col+GC
        # CB
        self.colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
        self.colorwheel[col:col+CB, 2] = 255
        col = col+CB
        # BM
        self.colorwheel[col:col+BM, 2] = 255
        self.colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
        col = col+BM
        # MR
        self.colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
        self.colorwheel[col:col+MR, 0] = 255
    
    def flow_to_img(self, flow, clip_flow=None):
        if clip_flow is not None:
            flow = np.clip(flow, 0, clip_flow)
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)

        flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.float32)
        ncols = self.colorwheel.shape[0]
        rad = np.sqrt(np.square(u) + np.square(v))
        a = np.arctan2(-v, -u) / np.pi
        fk = (a+1) / 2*(ncols-1)
        k0 = np.floor(fk).astype(np.int32)
        k1 = k0 + 1
        k1[k1 == ncols] = 0
        f = fk - k0
        for i in range(self.colorwheel.shape[1]):
            tmp = self.colorwheel[:,i]
            col0 = tmp[k0] / 255.0
            col1 = tmp[k1] / 255.0
            col = (1-f)*col0 + f*col1
            idx = (rad <= 1)
            col[idx]  = 1 - rad[idx] * (1-col[idx])
            col[~idx] = col[~idx] * 0.75
            flow_image[:,:,i] = col

        return flow_image 
        
    def __getitem__(self, index):
        folder_path = self.folder_paths[index]
        npy_paths = self.data_dict[folder_path]
        magnitude = self.magnitude_dict[os.path.basename(folder_path[:-1])]
        
        # Randomly sample image sequence based on magnitude
        magnitude_window_avg = [np.average(magnitude[i-2:i+2])\
             for i in range(2, len(magnitude)-2)]
        magnitude_window_avg = [0, 0] + magnitude_window_avg + [0, 0]
        magnitude_window_avg = magnitude_window_avg / np.sum(magnitude_window_avg)
        random_idx = np.random.choice(len(npy_paths), p=magnitude_window_avg)
        
        # Omit sequence with close distance (SSD)
        flows = []
        for i in range(random_idx-2, random_idx+3):
            flows.append(np.load(npy_paths[i]))
        cond1 = np.sum(np.power(flows[2] - flows[0], 2)) < self.min_distance
        cond2 = np.sum(np.power(flows[2] - flows[4], 2)) < self.min_distance
        if cond1 or cond2:
            return None, None
        
        # Randomly choose if positive or negative
        imgs = []
        # Positive sequence
        if np.random.rand() < self.positive_ratio:
            for i in range(1, 4):
                imgs.append(self.flow_to_img(flows[i]))
            lbl = 1
        # Negative sequence
        else:
            imgs.append(self.flow_to_img(flows[1]))
            # Randomly select the idx-2 frame or the idx+2 frame
            if np.random.rand() < 0.5:
                imgs.append(self.flow_to_img(flows[0]))
            else:
                imgs.append(self.flow_to_img(flows[4]))
            imgs.append(self.flow_to_img(flows[3]))
            lbl = 0
            
        # Randomly flip the order
        if np.random.rand() < 0.5:
            imgs = imgs[::-1]
        
        if self.transform:
            imgs = torch.stack([self.transform(img) for img in imgs])
        
        return imgs, lbl

    def __len__(self):
        return len(self.folder_paths)
    
def gymnastics_flow_collate(batch):
    imgs = torch.stack([item[0] for item in batch if item[0] is not None])
    lbl = torch.stack([torch.tensor(item[1]) for item in batch if item[0] is not None])
    return [imgs, lbl]
