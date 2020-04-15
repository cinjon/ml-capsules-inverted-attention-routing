import os
import json
import time
import shutil
import getpass
import numpy as np
from glob import glob

import torch

# NOTE: DON'T USE REALLY OBSOLETED
class GymnasticsFlow(torch.utils.data.Dataset):
    def __init__(self, root, bad_list_path,nan_list_path, magnitude_list_path,
                 file_dict_path, range_size=1, min_distance=0,
                 positive_ratio=0.25, transform=None):

        self.root = root
        self.bad_list_path = bad_list_path
        self.nan_list_path = nan_list_path
        self.magnitude_list_path = magnitude_list_path
        self.file_dict_path = file_dict_path
        self.range_size = range_size
        self.min_distance = min_distance
        self.positive_ratio = positive_ratio
        self.transform = transform

        self.get_paths()
        self.get_magnitude_dict()
        self.colorwheel = make_colorwheel()

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
        with open(self.file_dict_path) as f:
            self.file_dict = json.load(f)
        self.folder_paths = [
            folder_path for folder_path in list(self.file_dict.keys())\
                if os.path.basename(folder_path[:-1]) not in bad_flows_set]
        self.data_dict = {folder_path: self.file_dict[folder_path]\
            for folder_path in self.folder_paths}
        # self.folder_paths = [
        #     folder_path for folder_path in sorted(glob(os.path.join(self.root, "*/")))\
        #         if os.path.basename(folder_path[:-1]) not in bad_flows_set]
        # self.data_dict = {}
        # for folder_path in self.folder_paths:
        #     self.data_dict[folder_path] = (sorted(glob(os.path.join(folder_path, "*.npy"))))

    def get_magnitude_dict(self):
        self.magnitude_dict = {}
        magnitude = np.load(self.magnitude_list_path, allow_pickle=True)
        for m in magnitude:
            self.magnitude_dict[m[0]] = m[1]

    def __getitem__(self, index):
        folder_path = self.folder_paths[index]
        npy_paths = self.data_dict[folder_path]
        magnitude = self.magnitude_dict[os.path.basename(folder_path[:-1])]
        magnitude = np.nan_to_num(magnitude)

        # Randomly sample image sequence based on magnitude
        # magnitude_window_avg = [np.average(magnitude[i-2:i+3])\
        #      for i in range(2, len(magnitude)-2)]
        # magnitude_window_avg = [0, 0] + magnitude_window_avg + [0, 0]
        # magnitude_window_avg = magnitude_window_avg / np.sum(magnitude_window_avg)
        magnitude_window_avg = [np.average(magnitude[i-self.range_size*2:i+self.range_size+1])\
            for i in range(self.range_size*2, len(magnitude)-self.range_size*2)]
        magnitude_window_avg = [0] * (self.range_size*2) + magnitude_window_avg + [0] * (self.range_size*2)
        magnitude_window_avg = magnitude_window_avg / np.sum(magnitude_window_avg)
        if np.sum(np.isnan(magnitude_window_avg)) > 0:
            return None, None
        random_idx = np.random.choice(len(npy_paths), p=magnitude_window_avg)

        # Omit sequence with close distance (SSD)
        flows = []
        # for i in range(random_idx-2, random_idx+3):
        #     flows.append(np.load(npy_paths[i]))
        # cond1 = np.sum(np.power(flows[2] - flows[0], 2)) < self.min_distance
        # cond2 = np.sum(np.power(flows[2] - flows[4], 2)) < self.min_distance
        # if cond1 or cond2:
        #     return None # , None
        for i in range(random_idx-self.range_size*2, random_idx+self.range_size*2+1):
            flows.append(np.load(npy_paths[i]))

        '''Is handled by capsule_time_model.get_reorder_loss'''
        # # Randomly choose if positive or negative
        # imgs = []
        # # Positive sequence
        # if np.random.rand() < self.positive_ratio:
        #     for i in range(1, 4):
        #         imgs.append(self.flow_to_img(flows[i]))
        #     lbl = 1
        # # Negative sequence
        # else:
        #     imgs.append(self.flow_to_img(flows[1]))
        #     # Randomly select the idx-2 frame or the idx+2 frame
        #     if np.random.rand() < 0.5:
        #         imgs.append(self.flow_to_img(flows[0]))
        #     else:
        #         imgs.append(self.flow_to_img(flows[4]))
        #     imgs.append(self.flow_to_img(flows[3]))
        #     lbl = 0
        #
        # # Randomly flip the order
        # if np.random.rand() < 0.5:
        #     imgs = imgs[::-1]

        imgs = [flow_to_img(flow, self.colorwheel) for flow in flows]

        if self.transform:
            imgs = torch.stack([self.transform(img) for img in imgs])

        # It doesn't matter since the labels are handled in get_reorder_loss
        lbl = 0

        return imgs, lbl

    def __len__(self):
        return len(self.folder_paths)

def gymnastics_flow_collate(batch):
    imgs = torch.stack([item[0] for item in batch if item[0] is not None])
    lbl = torch.stack([torch.tensor(item[1]) for item in batch if item[0] is not None])
    return [imgs, lbl]


# Note: this one is fine
class GymnasticsFlowExperiment(torch.utils.data.Dataset):
    def __init__(self, root, file_dict_path, video_names, transform=None,
                 train=True, range_size=5, positive_ratio=0.5, is_flow=True):
        self.root = root
        self.file_dict_path = file_dict_path
        self.video_names = [n + '.fps25' for n in video_names.split(',')]
        self.transform = transform
        self.train = train
        self.range_size = range_size
        self.positive_ratio = positive_ratio
        self.is_flow = is_flow
        self.colorwheel = make_colorwheel()

        # Copying files
        print('Copying the files over ...', self.video_names)
        user = getpass.getuser()
        folder_name = 'flow' if self.is_flow else 'flow_imgs'
        path = '/scratch/%s/gymnastics/%s' % (user, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        for video_name in self.video_names:
            new_path = os.path.join(path, video_name)
            original_path = os.path.join(root, video_name)
            if not os.path.exists(new_path):
                shutil.copytree(original_path, new_path)
        self.root = path
        print('Done copying the files over.')

        # Get file dict
        with open(self.file_dict_path) as f:
            self.file_dict = json.load(f)
        self.file_dict = {video_name: self.file_dict[video_name]\
            for video_name in self.video_names}

        # Get determinstic indices
        self.indices = []
        for video_name in self.video_names:
            split = len(self.file_dict[video_name]) - 5*self.range_size
            if self.train:
                start = 0
                end = int(split * 4 / 5)
            else:
                start = int(split * 4 / 5)
                end = len(self.file_dict[video_name]) - 4*self.range_size
            for i in range(start, end):
                self.indices.append([video_name, i])

    def __getitem__(self, index):
        video_name, video_index = self.indices[index]
        npy_names = self.file_dict[video_name]
        npy_paths = [os.path.join(self.root, video_name, npy_names[i])\
            for i in range(video_index, video_index+self.range_size*5, self.range_size)]

        # Randomly choose if positive or negative
        # Positive sequence
        if np.random.rand() < self.positive_ratio:
            flows = [save_np_load(npy_paths[i]) for i in range(1, 4)]
            lbl = 1.
        # Negative sequence
        else:
            flows = [save_np_load(npy_paths[1])]
            # Randomly select the idx-2 frame or the idx+2 frame
            if np.random.rand() < 0.5:
                flows.append(save_np_load(npy_paths[0]))
            else:
                flows.append(save_np_load(npy_paths[4]))
            flows.append(save_np_load(npy_paths[3]))
            lbl = 0.

        # Randomly flip the order
        if np.random.rand() < 0.5:
            flows = flows[::-1]

        # Check whether flow_img or flow
        imgs = [flow_to_img(flow, self.colorwheel) for flow in flows]\
            if self.is_flow else flows

        if self.transform:
            imgs = torch.stack([self.transform(img) for img in imgs])

        return imgs, lbl

    def __len__(self):
        return len(self.indices)


########################################################################
# Optimal Flow visualization
########################################################################

def make_colorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def flow_to_img(flow, colorwheel, clip_flow=None):
    if clip_flow is not None:
        flow = np.clip(flow, 0, clip_flow)
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75
        flow_image[:,:,i] = np.floor(255 * col) # col

    return flow_image

# Convert all nan to zero
def save_np_load(path):
    return np.nan_to_num(np.load(path))
