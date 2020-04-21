import random

import h5py
import numpy as np
import torch
import torch.utils.data as data


class MovingMNIST(data.Dataset):
    def __init__(self, train=True, seq_len=20, skip=1,
                 image_size=64, colored=True, tiny=False,
                 is_triangle_loss=False):
        self.is_triangle_loss = is_triangle_loss
        handler = _ColoredBouncingMNISTDataHandler if colored else \
            _BouncingMNISTDataHandler
        self.data_handler = handler(seq_len, skip, image_size,
                                    is_triangle_loss=is_triangle_loss)
        if tiny:
            self.data_size = 64 * pow(2, 2)
        elif train:
            # self.data_size = 64 * pow(2, 12)
            # NOTE: I changed this so that we can get more epochs.
            self.data_size = 64 * pow(2, 8)
        else:
            self.data_size = 64 * pow(2, 5)

    def __getitem__(self, index):
        return torch.from_numpy(self.data_handler.get_item()), index

    def __len__(self):
        return self.data_size


class _BouncingMNISTDataHandler(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, seq_length = 20, skip=1, output_image_size=64,
                 is_triangle_loss=False):
        self.seq_length_ = seq_length
        self.skip = skip
        self.image_size_ = 64
        self.output_image_size = output_image_size
        self.num_digits_ = 2
        self.step_length_ = 0.035 # NOTE: was 0.1

        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2

        with h5py.File('/misc/kcgscratch1/ChoGroup/resnick/vidcaps/MovingMNist/mnist.h5', 'r') as f:
            self.data_ = f['train'][()].reshape(-1, 28, 28)

        self.indices_ = np.arange(self.data_.shape[0])
        self.row_ = 0
        self.is_triangle_loss = is_triangle_loss
        np.random.shuffle(self.indices_)

    def get_dims(self):
        return self.frame_size_

    def get_seq_length(self):
        return self.seq_length_

    def get_random_trajectory(self, num_digits):
        length = self.seq_length_
        skip = self.skip
        canvas_size = self.image_size_ - self.digit_size_

        # Initial position uniform random inside the box.
        y = np.random.rand(num_digits)
        x = np.random.rand(num_digits)

        # Choose a random velocity.
        theta = np.random.rand(num_digits) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, num_digits))
        start_x = np.zeros((length, num_digits))
        for i in range(length * skip):
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
                         self.image_size_, self.image_size_), dtype=np.float32)

        for n in range(self.num_digits_):

            # get random digit from dataset
            ind = self.indices_[self.row_]
            self.row_ += 1
            if self.row_ == self.data_.shape[0]:
                self.row_ = 0
                np.random.shuffle(self.indices_)
            digit_image = self.data_[ind, :, :]

            # generate video
            for i in range(self.seq_length_):
                top = start_y[i, n]
                left = start_x[i, n]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                data[i, top:bottom, left:right] = self.overlap(
                    data[i, top:bottom, left:right], digit_image)

        if self.output_image_size == self.image_size_:
            return data
        else:
            return self.resize(data, self.output_image_size)


class _ColoredBouncingMNISTDataHandler(_BouncingMNISTDataHandler):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def get_item(self, verbose=False):
        start_y, start_x = self.get_random_trajectory(self.num_digits_)

        # minibatch data
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
                np.random.shuffle(self.indices_)
            digit_image = self.data_[ind, :, :]

            # generate video
            for i in range(self.seq_length_):
                top = start_y[i, n]
                left = start_x[i, n]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                data[i, top:bottom, left:right, n] = self.overlap(
                    data[i, top:bottom, left:right, n], digit_image)

        data = np.transpose(data, (0, 3, 1, 2))
        # These come out as [seq_len, 3, 64, 64]
        if self.is_triangle_loss:
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
            return data
        else:
            ret = self.resize(data, self.output_image_size)
            return ret
