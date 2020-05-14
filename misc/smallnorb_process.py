import os
import gzip
import requests
import argparse
import numpy as np

import torch

IMG_NUM = 24300 * 2
IMG_SIZE = 96
TRAIN_SUFFIX = 'smallnorb-5x46789x9x18x6x2x96x96-'
VAL_SUFFIX = 'smallnorb-5x01235x9x18x6x2x96x96-'

resources = {
    'train': ['https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz',
              'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz',
              'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz'],
    'val':['https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz',
           'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz',
           'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz']
}

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, help='data root')
parser.add_argument(
    '--resize_shape', default=48, type=int, help='SmallNORB image resize shape')
args = parser.parse_args()

def download(root, train):
    os.makedirs(root, exist_ok=True)
    urls = resources['train'] if train else resources['val']
    for url in urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(root, filename)
        if os.path.exists(filepath):
            continue

        print('Downloading %s' % url)
        myfile = requests.get(url)
        open(filepath, 'wb').write(myfile.content)

def get_imgs(root, train):
    file_path = os.path.join(root, TRAIN_SUFFIX+'training-dat.mat.gz') if train\
        else os.path.join(root, VAL_SUFFIX+'testing-dat.mat.gz')

    imgs = np.zeros((IMG_NUM, IMG_SIZE ** 2))
    with gzip.open(file_path, 'rb') as f:
        for i in range(6):
            _ = f.read(4)

        for i in range(IMG_NUM):
            imgs[i] = np.frombuffer(f.read(IMG_SIZE ** 2), 'uint8')

    return imgs.reshape((-1, IMG_SIZE, IMG_SIZE))

def get_lbls(root, train):
    file_path = os.path.join(root, TRAIN_SUFFIX+'training-cat.mat.gz') if train\
        else os.path.join(root, VAL_SUFFIX+'testing-cat.mat.gz')

    with gzip.open(file_path, 'rb') as f:
        for i in range(5):
            _ = f.read(4)

        lbls = np.frombuffer(f.read(
            IMG_NUM * np.dtype('int32').itemsize), 'int32')

    return lbls.repeat(2, axis=0)

def get_info(root, train):
    file_path = os.path.join(root, TRAIN_SUFFIX+'training-info.mat.gz') if train\
        else os.path.join(root, VAL_SUFFIX+'testing-info.mat.gz')

    with gzip.open(file_path, 'rb') as f:
        for i in range(5):
            _ = f.read(4)

        info = np.frombuffer(f.read(
            4 * IMG_NUM // 2 * np.dtype('int32').itemsize),
            'int32')

        info = info.reshape(IMG_NUM // 2, 4)

        elevation = np.array([30,35,40,45,50,55,60,65,70])

        info = info.repeat(2, axis=0)

    return info

def process_data(root, train, resize_shape=48):
    download(root, train)

    imgs = get_imgs(root, train)
    lbls = get_lbls(root, train)
    info = get_info(root, train)

    img_sequences = []
    lbl_sequences = []
    info_sequences = []

    instances = [4, 6, 7, 8, 9] if train else [0, 1, 2, 3, 5]

    for lbl in range(5):
        for instance in instances:
            for elevation in range(9):
                for lighting in range(6):
                    index = np.where(
                        (lbls == lbl) & (info[:, 0] == instance) &\
                        (info[:, 1] == elevation) & (info[:, 3] == lighting))[0]
                    for i in range(2):
                        temp_index = index[i::2]
                        temp_azimuth = info[:, 2][temp_index]
                        temp_index = temp_index[np.argsort(temp_azimuth)]

                        img_sequences.append(imgs[temp_index])
                        lbl_sequences.append(lbl)

                        temp_info = np.array(
                            [[instance, elevation, 0, lighting],] * len(temp_index))
                        temp_info[:, 2] = list(range(0, 36, 2))
                        info_sequences.append(temp_info)

    img_sequences = np.array(img_sequences)
    lbl_sequences = np.array(lbl_sequences)
    info_sequences = np.array(info_sequences)

    # Resize
    sequence_num = img_sequences.shape[0]
    img_sequences = img_sequences / 255.
    img_sequences = torch.from_numpy(img_sequences).view(-1, 1, IMG_SIZE, IMG_SIZE)
    img_sequences = torch.nn.Upsample((resize_shape, resize_shape), mode='bilinear', align_corners=True)(
        img_sequences).view(-1, resize_shape * resize_shape)

    # Per sample normalization
    img_sequences = (img_sequences - img_sequences.mean(1).unsqueeze(-1)) /\
        img_sequences.std(1).unsqueeze(1)
    img_sequences = img_sequences.view(sequence_num, -1, resize_shape, resize_shape)

    split_name = 'train' if train else 'val'
    np.save(os.path.join(root, 'smallnorb_%s_imgs.npy' % split_name), img_sequences.numpy())
    np.save(os.path.join(root, 'smallnorb_%s_lbls.npy' % split_name), lbl_sequences)
    np.save(os.path.join(root, 'smallnorb_%s_info.npy' % split_name), info_sequences)

if __name__ == '__main__':
    process_data(args.root, True, args.resize_shape)
    process_data(args.root, False, args.resize_shape)
