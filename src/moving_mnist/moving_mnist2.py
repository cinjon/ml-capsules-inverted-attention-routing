from collections import defaultdict
import copy
import gzip
import math
import os
import random

import h5py
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, makedir_exist_ok
from torchvision.transforms import RandomResizedCrop, ColorJitter, RandomApply, \
    Compose, RandomHorizontalFlip, ToPILImage, ToTensor


class MovingMNISTClassSampler(data.Sampler):
    def __init__(self, data_source, num_replicas=None, rank=None, shuffle=True):
        self.data_source = data_source
        self.epoch = 0
        self.shuffle = shuffle

        index_by_label = defaultdict(list)
        for index, label in enumerate(data_source.labels):
            index_by_label[int(label)].append(index)
        self.index_by_label = index_by_label        
        self.total_num_samples = len(data_source.labels)
        self.num_replicas = num_replicas
        self.num_samples = int(
            math.ceil(len(data_source) * 1.0 / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.rank = rank

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        index_by_label = copy.copy(self.index_by_label)
        if self.shuffle:
            random.seed(self.epoch)
            for index, lst in index_by_label.items():
                random.shuffle(lst)
        
        indices = []
        row = 0
        while len(indices) < self.total_num_samples:
            for index in sorted(index_by_label.keys()):
                if row >= len(index_by_label[index]):
                    continue
                indices.append(index_by_label[index][row])
            row += 1

        # add extra samples to make it evenly divisible
        print("Len ndices: ", len(indices))
        print(indices[:25], indices[-25:])
        if self.total_size > len(indices):
            indices += indices[:(self.total_size - len(indices))]
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class MovingMNIST(data.Dataset):
    def __init__(self, root, train=True, seq_len=20, skip=1,
                 image_size=64, colored=True, tiny=False,
                 is_three_images=False, is_reorder_loss=False,
                 num_digits=2, one_data_loop=False,
                 center_start=False, step_length=0.035,
                 positive_ratio=0.5, single_angle=False,
                 use_simclr_xforms=False,
                 use_diff_class_digit=False):
        self.root = root
        self.is_three_images = is_three_images
        self.is_reorder_loss = is_reorder_loss
        self.use_simclr_xforms = train and use_simclr_xforms
        # If use_diff_class_digit, then we use a diff digit from the same class
        # for every position.
        self.use_diff_class_digit = use_diff_class_digit
        if self.use_simclr_xforms:
            self.xforms = Compose([
                ToPILImage(),
                RandomResizedCrop(image_size, scale=(0.8, 1.0), # NOTE: was (0.08, 1)
                                  ratio=(0.75, 1.333), interpolation=2),
                RandomHorizontalFlip(p=0.5),
                # RandomApply([ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                ToTensor()
            ])

        data, labels = self.load_dataset(train)
        self.labels = labels

        handler = _ColoredBouncingMNISTDataHandler if colored else \
            _BouncingMNISTDataHandler
        self.data_handler = handler(
            data, labels, seq_len, skip, image_size,
            is_three_images=is_three_images,
            is_reorder_loss=is_reorder_loss,
            num_digits=num_digits, step_length=step_length,
            train=train, center_start=center_start, single_angle=single_angle,
            positive_ratio=positive_ratio,
            use_diff_class_digit=use_diff_class_digit
        )

        if one_data_loop:
            self.data_size = len(data)
        elif tiny:
            self.data_size = 64 * pow(2, 2)
        elif train:
            # self.data_size = 64 * pow(2, 12)
            # NOTE: I changed this so that we can get more epochs.
            self.data_size = 64 * pow(2, 9) # NOTE: Has been 8 for a long time.
        else:
            self.data_size = 64 * pow(2, 5)

    def load_dataset(self, train):
        img_filename = "train-images-idx3-ubyte.gz" if train\
            else "t10k-images-idx3-ubyte.gz"
        lbl_filename = "train-labels-idx1-ubyte.gz" if train\
            else "t10k-labels-idx1-ubyte.gz"

        # Download data if not exist
        # root = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/MovingMNist/'
        makedir_exist_ok(self.root)
        img_filepath = self.download(self.root, img_filename)
        lbl_filepath = self.download(self.root, lbl_filename)

        # Load image data
        with gzip.open(img_filepath, "rb") as f:
            img_data = np.frombuffer(f.read(), np.uint8, offset=16)
        img_data = img_data.reshape(-1, 1, 28, 28)
        img_data = img_data.astype(np.float) / 255.

        # Load label data
        with gzip.open(lbl_filepath, "rb") as f:
            lbl_data = np.frombuffer(f.read(), np.uint8, offset=8)
        lbl_data = lbl_data.astype(np.int)
        return img_data, lbl_data

    @staticmethod
    def download(root, filename):
        filepath = os.path.join(root, filename)
        if not os.path.exists(filepath):
            print("http://yann.lecun.com/exdb/mnist/" + filename)
            download_url("http://yann.lecun.com/exdb/mnist/" + filename,
                         root=root)
        return filepath

    def __getitem__(self, index):
        datum, label = self.data_handler.get_item()
        # datum is [3 1, 64, 64]
        if self.use_simclr_xforms:
            datum = transform(datum, self.xforms)
        else:
            datum = torch.from_numpy(datum)
        label = torch.from_numpy(np.array(label))
        return datum, label

    def __len__(self):
        return self.data_size


def transform(arr, transforms):
    tensors = []
    for npimg in arr:
        npimg = npimg.transpose((1, 2, 0))
        tensor = transforms(npimg)
        tensors.append(tensor)
    return torch.stack(tensors)


class _BouncingMNISTDataHandler(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, data, labels, seq_length=20, skip=1,
                 output_image_size=64, is_three_images=False,
                 is_reorder_loss=False, num_digits=2,
                 step_length=0.035, train=False, center_start=False,
                 single_angle=False, positive_ratio=0.5, use_diff_class_digit=False):
        self.seq_length_ = seq_length
        self.skip = skip
        self.image_size_ = 64
        self.output_image_size = output_image_size
        self.num_digits_ = num_digits
        self.step_length_ = step_length # NOTE: was 0.1

        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2

        self.data_ = data
        self.labels_ = labels

        self.indices_ = np.arange(self.data_.shape[0])

        self.row_ = 0
        self.is_three_images = is_three_images
        self.is_reorder_loss = is_reorder_loss
        self.train = train
        self.center_start = center_start
        self.single_angle = single_angle
        self.positive_ratio = positive_ratio
        if train:
            np.random.shuffle(self.indices_)

        self.use_diff_class_digit = use_diff_class_digit
        if use_diff_class_digit:
            index_by_label = defaultdict(list)
            self.index_rows = {}
            for index in self.indices_:
                label = labels[index]
                index_by_label[int(label)].append(index)
                self.index_rows[int(label)] = 0
            self.index_by_label = index_by_label

    def get_dims(self):
        return self.frame_size_

    def get_seq_length(self):
        return self.seq_length_

    def get_random_trajectory(self, num_digits):
        length = self.seq_length_
        skip = self.skip
        canvas_size = self.image_size_ - self.digit_size_

        if self.center_start:
            # NOTE: Trying deterministic here
            x = [0.5] * num_digits
            y = [0.5] * num_digits
        else:
            # Initial position uniform random inside the box.
            y = np.random.rand(num_digits)
            x = np.random.rand(num_digits)

        # Choose a random velocity.
        if self.single_angle:
            theta = np.pi / 2.
        else:
            theta = np.random.rand(num_digits) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, num_digits))
        start_x = np.zeros((length, num_digits))
        start_y[0] = y
        start_x[0] = x
        for i in range(1, length * skip):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            for j in range(num_digits):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]

            if not i % skip == 0:
                continue

            index = int(i / skip)
            start_y[index, :] = y
            start_x[index, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def overlap(self, a, b):
        """ Put b on top of a."""
        return np.maximum(a, b)
        # return b

    def resize(self, data, size):
        output_data = np.zeros((self.seq_length_, size, size),
                               dtype=np.float32)

        for i, frame in enumerate(data):
            output_data[i] = scipy.misc.imresize(frame, (size, size))

        return output_data

    def get_item(self, verbose=False):
        start_y, start_x = self.get_random_trajectory(self.num_digits_)

        # minibatch data
        data = np.zeros((self.seq_length_,
                         self.image_size_,
                         self.image_size_),
                        dtype=np.float32)

        label = []
        for n in range(self.num_digits_):

            # get random digit from dataset
            ind = self.indices_[self.row_]
            self.row_ += 1
            if self.row_ == self.data_.shape[0]:
                self.row_ = 0
                if self.train:
                    np.random.shuffle(self.indices_)

            digit_image = self.data_[ind, :, :]
            label.append(self.labels_[ind])

            # generate video
            labels_used = [label[-1]]                
            for i in range(self.seq_length_):
                if i > 0 and self.use_diff_class_digit:
                    index_label = int(label[-1])
                    label_indices = self.index_by_label[index_label]
                    if self.index_rows[index_label] == len(label_indices):
                        self.index_rows[index_label] = 0
                    index_row = self.index_rows[index_label]
                    next_data_index = label_indices[index_row]
                    digit_image = self.data_[next_data_index, :, :]
                    self.index_rows[index_label] += 1
                    labels_used.append(self.labels_[next_data_index])

                top = start_y[i, n]
                left = start_x[i, n]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                data[i, top:bottom, left:right] = self.overlap(
                    data[i, top:bottom, left:right], digit_image)

            # print('Yo: ', labels_used)

        data = data[:, None, :, :]
        # These come out as [seq_len, 3, 64, 64]
        if self.is_three_images:
            p = random.random()
            # Sample three from somewhere in here, but allow for jumps of two.
            if p > 0.75:
                data = data[:3]
            elif p > 0.5:
                data = data[1:4]
            elif p > 0.25:
                data = data[2:]
            else:
                data = data[[0, 2, 4]]
        elif self.is_reorder_loss:
            use_positive = random.random() < self.positive_ratio
            if use_positive:
                data = data[1:4]
                label = 1.
            else:
                if random.random() > 0.5:
                    data = np.stack([data[1], data[0], data[3]])
                else:
                    data = np.stack([data[1], data[4], data[3]])
                label = 0.

        if self.output_image_size == self.image_size_:
            ret = data
        else:
            ret = self.resize(data, self.output_image_size)

        return ret, label


class _ColoredBouncingMNISTDataHandler(_BouncingMNISTDataHandler):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def get_item(self, verbose=False):
        start_y, start_x = self.get_random_trajectory(self.num_digits_)

        # minibatch data
        label = []
        data = np.zeros((self.seq_length_,
                         self.image_size_,
                         self.image_size_,
                         3),
                        dtype=np.float32)

        for n in range(self.num_digits_):

            # get random digit from dataset
            ind = self.indices_[self.row_]
            self.row_ += 1
            if self.row_ == self.data_.shape[0]:
                self.row_ = 0
                if self.train:
                    np.random.shuffle(self.indices_)
            digit_image = self.data_[ind, :, :]
            label.append(self.labels_[ind])

            # generate video
            for i in range(self.seq_length_):
                if i > 0 and self.use_diff_class_digit:
                    index_label = int(self.labels_[ind])
                    if self.index_rows[index_label] == len(self.index_by_label[index_label]):
                        self.index_rows[index_label] = 0
                    index_row = self.index_rows[index_label]
                    digit_image = self.data_[index_row, :, :]
                    self.index_rows[index_label] += 1

                top = start_y[i, n]
                left = start_x[i, n]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                data[i, top:bottom, left:right, n] = self.overlap(
                    data[i, top:bottom, left:right, n], digit_image)

        data = np.transpose(data, (0, 3, 1, 2))
        # These come out as [seq_len, 3, 64, 64]
        if self.is_three_images:
            p = random.random()
            # Sample three from somewhere in here, but allow for jumps of two.
            if p > 0.75:
                data = data[:3]
            elif p > 0.5:
                data = data[1:4]
            elif p > 0.25:
                data = data[2:]
            else:
                data = data[[0, 2, 4]]

        if self.output_image_size == self.image_size_:
            ret = data
        else:
            ret = self.resize(data, self.output_image_size)

        return ret, label
