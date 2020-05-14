import os
import numpy as np

import torch
import torchvision.transforms as transforms

class SmallNORB(torch.utils.data.Dataset):
    def __init__(self, root, train=True, num_frames=3,
                 min_brightness_factor=-0.5, max_brightness_factor=0.5,
                 min_contrast_factor=0.8, max_contrast_factor=1.2):
        self.root = root
        self.train = train
        self.num_frames=3
        self.min_brightness_factor = min_brightness_factor
        self.max_brightness_factor = max_brightness_factor
        self.min_contrast_factor = min_contrast_factor
        self.max_contrast_factor = max_contrast_factor

        if self.train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                # transforms.ColorJitter(
                #     brightness=(0, 2), contrast=(0.5, 1.5), saturation=0, hue=0),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(32),
                transforms.ToTensor()
            ])

        split_name = 'train' if self.train else 'val'
        self.imgs = np.load(
            os.path.join(root, 'smallnorb_%s_imgs.npy' % split_name))
        # self.imgs = np.uint8(self.imgs * 255.)
        self.imgs = np.float32(self.imgs)

        self.lbls = np.load(
            os.path.join(root, 'smallnorb_%s_lbls.npy' % split_name))

        # self.info = np.load(
        #     os.path.join(root, 'smallnorb_%s_info.npy' % split_name))

        self.num_sequence = self.imgs.shape[1] - self.num_frames + 1

    def __getitem__(self, index):
        vid_id = index // self.num_sequence
        first_frame_id = index - vid_id * self.num_sequence

        img_sequence = []
        for i in range(self.num_frames):
            brightness_factor = np.random.uniform(
                self.min_brightness_factor, self.max_brightness_factor)
            contrast_factor = np.random.uniform(
                self.min_contrast_factor, self.max_contrast_factor)

            img = self.transform(self.imgs[vid_id, first_frame_id+i])
            img = img * contrast_factor + brightness_factor
            img_sequence.append(img)
        img_sequence = torch.stack(img_sequence)

        return img_sequence, self.lbls[vid_id]

    def __len__(self):
        return self.imgs.shape[0] * self.num_sequence
