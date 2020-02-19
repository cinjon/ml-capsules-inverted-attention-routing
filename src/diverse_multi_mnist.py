from collections import defaultdict
import os
import random

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import IterableDataset
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, \
    makedir_exist_ok, verify_str_arg
from torchvision.datasets.mnist import read_sn3_pascalvincent_tensor, read_image_file, read_label_file
from torchvision.transforms import Compose, Normalize, ToTensor


class DiverseMultiMNist(IterableDataset):
    resources = [("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                  "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
                 ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                  "d53e105ee54ea40749a09fcbcd1e9432"),
                 ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                  "9fb629c4189551a2d022fa330f9573f3"),
                 ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                  "ec29112dd5afa0611ce80d1b7f02629c")]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = [
        '0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five',
        '6 - six', '7 - seven', '8 - eight', '9 - nine'
    ]

    def __init__(self, root, train=True, download=False):
        super(DiverseMultiMNist, self).__init__()
        self.root = root
        self.train = train
        if download:
            self.download()

        data_file = self.training_file if train else self.test_file
        self.data, self.targets = torch.load(
            os.path.join(self.processed_folder, data_file))
        self.num_images = len(self.data)
        print('Number of images: ', self.num_images)
        data_by_label = defaultdict(list)
        for datum, target in zip(self.data, self.targets):
            data_by_label[target.item()].append(datum)
        self.data_by_label = data_by_label

        del self.targets
        del self.data

        self.transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self):
        return (os.path.exists(
            os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(
                    os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url,
                                         download_root=self.raw_folder,
                                         filename=filename,
                                         md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (read_image_file(
            os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
                        read_label_file(
                            os.path.join(self.raw_folder,
                                         'train-labels-idx1-ubyte')))
        test_set = (read_image_file(
            os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
                    read_label_file(
                        os.path.join(self.raw_folder,
                                     't10k-labels-idx1-ubyte')))
        with open(os.path.join(self.processed_folder, self.training_file),
                  'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file),
                  'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        print('Worker Info: ', worker_info)
        return self

    def __next__(self):
        """
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target1, target2 = np.random.choice(10, size=2, replace=False)

        index1 = np.random.choice(len(self.data_by_label[target1]), size=1)[0]
        index2 = np.random.choice(len(self.data_by_label[target2]), size=1)[0]
        img1 = self.data_by_label[target1][index1].numpy()
        img2 = self.data_by_label[target2][index2].numpy()

        # Random translations.
        tx1, tx2, ty1, ty2 = np.random.choice(range(-4, 5),
                                              size=4,
                                              replace=True)
        h, w = img1.shape
        padded_img1 = np.zeros((h + 8, w + 8), img1.dtype)
        padded_img1[4 - tx1:h + 4 - tx1, 4 - ty1:w + 4 - ty1] = img1
        padded_img2 = np.zeros((h + 8, w + 8), img2.dtype)
        padded_img2[4 - tx2:h + 4 - tx2, 4 - ty2:w + 4 - ty2] = img2

        target = torch.zeros(10)
        target_indices = [target1]
        if random.random() < 1. / 6:
            img = padded_img1.astype(np.uint8)
        else:
            img = (0.5 * padded_img1 + 0.5 * padded_img2).astype(np.uint8)
            target_indices.append(target2)

        target_indices = torch.tensor(target_indices)
        target.scatter_(0, target_indices, 1.)

        img = Image.fromarray(img, mode='L')
        img = self.transforms(img)
        # Repeat so that we have a 3 channel RGB image.
        img = img.repeat(3, 1, 1)
        return img, target

    def __len__(self):
        return self.num_images
