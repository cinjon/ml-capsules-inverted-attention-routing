import os
import gzip
import math
import random
import numpy as np
from PIL import Image
from glob import glob
from scipy.io.matlab import loadmat

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, makedir_exist_ok


class NewMovingMNist(Dataset):

    def __init__(self,
                 root,
                 train=True,
                 dataset_size=120000,
                 sequence=True,
                 width=64,
                 height=64,
                 num_frames=20,
                 original_size=28,
                 nums_per_image=2,
                 chance_single_label=0.2):

        self.root = root
        self.dataset_size = dataset_size
        self.train = train
        self.sequence = sequence
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.original_size = original_size
        self.nums_per_image = nums_per_image
        self.chance_single_label = chance_single_label

        self._transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1397,), (0.3081,))])

        if self.train:
            self.load_dataset()

            self.indices_list = []
            for i in range(10):
                indices = np.argwhere(self.lbl_data == i).flatten()
                self.indices_list.append(indices)

            # Get how many pixels can we move around a single image
            self.x_lim = self.width - self.original_size
            self.y_lim = self.height - self.original_size
            self.lims = (self.x_lim, self.y_lim)
        else:
            self.lbls = np.load(os.path.join(root, "moving_mnist_lbl_val.npy"))

            self.img_folder = os.path.join(root, "val_imgs")

    def __getitem__(self, index):
        # Train set
        if self.train:
            # Randomly choose single-label image or multi-label image
            if np.random.rand() > self.chance_single_label:
                this_nums_per_image = self.nums_per_image
            else:
                this_nums_per_image = 1

            # Create a np array of shape of
            # num_frames x 1 x new_width x new_height
            # Eg: 20 x 1 x 64 x 64
            _imgs = np.empty((self.num_frames, 1, self.width, self.height),
                             dtype=np.uint8)
            lbls = np.zeros((10,), dtype=np.float32)

            # Randomly generate direction, speed and velocity
            # for both images
            direcs = np.pi * (np.random.rand(this_nums_per_image) * 2 - 1)
            speeds = np.random.randint(5, size=this_nums_per_image) + 2
            veloc = np.asarray(
                [(speed * math.cos(direc), speed * math.sin(direc))\
                    for direc, speed in zip(direcs, speeds)])

            # Pick random categories
            random_categories = np.random.choice(range(10),
                                                 this_nums_per_image,
                                                 replace=False)

            # Assign labels
            lbls[random_categories] = 1.

            # Pick random images within those random categories
            # TODO: check if original paper avoid having same numbers
            # in an image
            random_indices = []
            for random_category in random_categories:
                random_indices.append(
                    np.random.choice(self.indices_list[random_category]))

            mnist_images = []
            positions = []
            for r in random_indices:
                # Get a list containing two PIL images randomly sampled
                # from the database
                img = get_image_from_array(self.img_data[r])
                img = Image.fromarray(img).resize(
                    (self.original_size, self.original_size), Image.ANTIALIAS)
                mnist_images.append(img)

                # Generate tuples of (x, y) i.e initial positions for
                # nums_per_image (default: 2)
                positions.append([
                    np.random.rand() * self.x_lim,
                    np.random.rand() * self.y_lim
                ])

            positions = np.asarray(positions)

            # Generate new frames for the entire num_frames
            for frame_idx in range(self.num_frames):
                canvases = [Image.new("L", (self.width, self.height))\
                    for _ in range(this_nums_per_image)]
                canvas = np.zeros((1, self.width, self.height),
                                  dtype=np.float32)

                # In canv (i.e Image object) place the image at the
                # respective positions
                # Super impose both images on the canvas
                # (i.e empty np array)
                for i, canv in enumerate(canvases):
                    canv.paste(mnist_images[i], tuple(positions[i].astype(int)))
                    canvas += arr_from_img(canv)

                    # Get the next position by adding velocity
                    next_pos = positions + veloc

                    # Iterate over velocity and see if we hit the wall
                    # If we do then change the direction
                    for i, pos in enumerate(next_pos):
                        for j, coord in enumerate(pos):
                            if coord < -2 or coord > self.lims[j] + 2:
                                veloc[i] = list(
                                    list(veloc[i][:j]) +\
                                    [-1 * veloc[i][j]] +\
                                    list(veloc[i][j+1:]))

                    # Make the permanent change to position by adding
                    # updated velocity
                    positions = positions + veloc

                    # Add the canvas to the dataset array
                    _imgs[frame_idx] =\
                        (canvas * 255.).clip(0, 255).astype(np.uint8)

            # Perform transformation
            imgs = torch.cat(
                [self._transforms(img[0]).unsqueeze(0) for img in _imgs])

            # TODO: change num_frames to 1 instead of this to save time
            if not self.sequence:
                imgs = imgs[0]

        # Validation set
        else:
            _imgs = np.load(
                os.path.join(self.img_folder,
                             "{}.npy".format(str(index).zfill(5))))
            imgs = torch.cat(
                [self._transforms(img[0]).unsqueeze(0) for img in _imgs])
            if not self.sequence:
                imgs = imgs[0]

            lbls = self.lbls[index]

        return imgs, lbls

    def __len__(self):
        if self.train:
            _len = self.dataset_size
        else:
            _len = len(self.lbls)
        return _len

    def load_dataset(self):
        img_filename = "train-images-idx3-ubyte.gz" if self.train\
            else "t10k-images-idx3-ubyte.gz"
        lbl_filename = "train-labels-idx1-ubyte.gz" if self.train\
            else "t10k-labels-idx1-ubyte.gz"

        # Download data if not exist
        makedir_exist_ok(self.root)
        img_filepath = self.download(img_filename)
        lbl_filepath = self.download(lbl_filename)

        # Load image data
        with gzip.open(img_filepath, "rb") as f:
            self.img_data = np.frombuffer(f.read(), np.uint8, offset=16)
        self.img_data = self.img_data.reshape(-1, 1, 28, 28)

        # Load label data
        with gzip.open(lbl_filepath, "rb") as f:
            self.lbl_data = np.frombuffer(f.read(), np.uint8, offset=8)

    def download(self, filename):
        filepath = os.path.join(self.root, filename)
        if not os.path.exists(filepath):
            print("http://yann.lecun.com/exdb/mnist/" + filename)
            download_url("http://yann.lecun.com/exdb/mnist/" + filename,
                         root=self.root)
        return filepath


class MovingMNist(Dataset):

    def __init__(self, root, train=True, sequence=False):
        self.root = root
        self.train = train
        self.sequence = sequence

        self._transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1397,), (0.3081,))])

        # Get file name
        if self.train:
            img_filename = "moving_mnist_img_train.npy"
            lbl_filename = "moving_mnist_lbl_train.npy"
        else:
            img_filename = "moving_mnist_img_val.npy"
            lbl_filename = "moving_mnist_lbl_val.npy"

        # Load data
        self.img_data = np.load(os.path.join(self.root,
                                             img_filename)).transpose(
                                                 (1, 0, 2, 3, 4))
        self.lbl_data = np.load(os.path.join(self.root,
                                             lbl_filename)).astype(np.float32)

        if self.sequence:
            self.seq_len = self.img_data.shape[1]
        else:
            self.img_data = self.img_data[:, 0]
            self.seq_len = 1

    def __getitem__(self, index):
        # Return a seuqnece
        if self.sequence:
            # Get random frame
            frame_num = random.choice(range(self.seq_len - 1))

            # Process frames
            this_frame = self._transforms(self.img_data[index, frame_num, 0])
            next_frame = self._transforms(self.img_data[index, frame_num + 1,
                                                        0])

            return this_frame, next_frame

        # Return the first frame in a sequence
        else:
            img = self._transforms(self.img_data[index, 0])
            lbl = self.lbl_data[index]
            return img, lbl

    def __len__(self):
        return len(self.img_data)


def MovingMNist_sequence_collate(batch):
    frame1 = torch.stack([item[0] for item in batch if item[2]])
    frame2 = torch.stack([item[1] for item in batch if item[2]])
    return [frame1, frame2]


class affNIST(Dataset):

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

        # self.transform = transforms.Compose([
        #     # transforms.ToPILImage(),
        #     # transforms.Pad(12),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1397,), (0.3081,))])

        self.imgs = []
        self.lbls = []
        for path in self.paths:
            data_arr = loadmat(path)["affNISTdata"][0, 0]
            self.lbls.append(data_arr[5].flatten())
            self.imgs.append(data_arr[2].transpose((1, 0)).reshape(
                (-1, 40, 40)))

        self.imgs = np.concatenate(self.imgs)
        self.lbls = np.concatenate(self.lbls)

    def __getitem__(self, index):
        img = self.imgs[index] if self.transform == None else self.transform(
            self.imgs[index])
        lbl = self.lbls[index]
        return img, lbl

    def __len__(self):
        return len(self.lbls)


def arr_from_img(img, mean=0, std=1):
    '''
    Args:
        im: Image
        shift: Mean to subtract
        std: Standard Deviation to subtract
    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.
    '''
    width, height = img.size
    arr = img.getdata()
    c = int(np.product(arr.size) / (width * height))
    return (np.asarray(arr, dtype=np.float32).reshape(
        (height, width, c)).transpose(2, 1, 0) / 255. - mean) / std


def get_image_from_array(X, mean=0, std=1):
    '''
    Args:
        X: Image of shape C x W x H
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    '''
    c, w, h = X.shape[0], X.shape[1], X.shape[2]
    ret = (((X + mean) * 255.) * std).reshape(c, w, h).transpose(2, 1, 0).clip(
        0, 255).astype(np.uint8)
    if c == 1:
        ret = ret.reshape(h, w)
    return ret
