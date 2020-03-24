import argparse
import gzip
import os
import math
import shutil
import sys
import time

import numpy as np
from PIL import Image

from torchvision.datasets.utils import download_url, makedir_exist_ok

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dest",
    default="",
    type=str,
    help="data dir")
parser.add_argument(
    "--mnist_folder",
    default="/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/data/MNIST/raw",
    type=str,
    help="data dir")
parser.add_argument(
    "--training",
    action='store_true',
    help="generate train set or val set")
parser.add_argument(
    "--frame_size",
    default=64,
    type=int,
    help="image size in each frame")
parser.add_argument(
    "--num_frames",
    default=20,
    type=int,
    help="number of frame in one sequence")
parser.add_argument(
    "--num_images",
    default=20000,
    type=int,
    help="number of sequences")
parser.add_argument(
    "--original_size",
    default=28,
    type=int,
    help="size of mnist digit within frame")
parser.add_argument(
    "--nums_per_image",
    default=2,
    type=int,
    help="number of mnist digit in one frame")
parser.add_argument(
    "--num_single_label",
    default=2,
    type=int,
    help="number of sequences with single digit")
args = parser.parse_args()


def arr_from_img(img, mean=0, std=1):
    '''
    Args:
        im: Image
        shift: Mean to subtract
        std: Standard Deviation to subtract
    Returns:
        Image in np.float32 format, in width height channel format. With values
            in range 0,1
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
    ret = (((X + mean) * 255.) * std).reshape(
        c, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if c == 1:
        ret = ret.reshape(h, w)
    return ret

def download(root, filename):
    filepath = os.path.join(root, filename)
    if not os.path.exists(filepath):
        download_url(
            "http://yann.lecun.com/exdb/mnist/" + filename,
            root=root)

    return filepath

def load_dataset(root, training=True):
    img_filename = "train-images-idx3-ubyte.gz" if training\
        else "t10k-images-idx3-ubyte.gz"
    lbl_filename = "train-labels-idx1-ubyte.gz" if training\
        else "t10k-labels-idx1-ubyte.gz"

    # Download data if not exist
    makedir_exist_ok(root)
    img_filepath = download(root, img_filename)
    lbl_filepath = download(root, lbl_filename)

    # Load image data
    with gzip.open(img_filepath, "rb") as f:
        imgs = np.frombuffer(f.read(), np.uint8, offset=16)
    imgs = imgs.reshape(-1, 1, 28, 28)

    # Load label data
    with gzip.open(lbl_filepath, "rb") as f:
        lbls = np.frombuffer(f.read(), np.uint8, offset=8)

    print('Wat: ', len(lbls), lbl_filename)
    print('Wat2: ', len(imgs), img_filename)

    return imgs, lbls

def generate_moving_mnist(
    mnist_folder, training, shape, num_frames, num_images,
    original_size, nums_per_image, num_single_label):
    width, height = shape
    img_data, lbl_data = load_dataset(mnist_folder, training)

    # Get label metadata in MNists
    repeat_num = math.ceil(nums_per_image * num_images / len(lbl_data))
    print("Repeat Num: ", repeat_num, nums_per_image, num_images)
    category_nums = np.zeros(10, dtype=np.int32)
    indices_list = []
    for i in range(10):
        indices = np.repeat(np.argwhere(lbl_data == i).flatten(), repeat_num)
        np.random.shuffle(indices)
        category_nums[i] += len(indices)
        indices_list.append(list(indices))

    # Get how many pixels can we move around a single image
    x_lim, y_lim = width - original_size, height - original_size
    lims = (x_lim, y_lim)

    # Create a dataset of shape of
    # num_frames x num_images x 1 x new_width x new_height
    # Eg: 20 x 10000 x 1 x 64 x 64
    imgs = np.empty(
        (num_frames, num_images, 1, width, height), dtype=np.uint8)
    lbls = np.zeros((num_images, 10), dtype=np.uint8)

    for img_idx in range(num_images):
        # Logging
        if (img_idx + 1) % 100 == 0:
            print("{}/{}".format(img_idx+1, num_images), end="\r")

        # Switch to single label
        if img_idx == num_images - num_single_label:
            nums_per_image = 1

        # Randomoly generate direction, speed and velocity
        # for both images
        direcs = np.pi * (np.random.rand(nums_per_image) * 2 - 1)
        speeds = np.random.randint(5, size=nums_per_image) + 2
        veloc = np.asarray(
            [(speed * math.cos(direc), speed * math.sin(direc))\
                for direc, speed in zip(direcs, speeds)])

        # Pick random categories
        random_categories = np.random.choice(
            range(10),
            nums_per_image,
            replace=False,
            p=category_nums/sum(category_nums))

        # Assign labels
        for random_category in random_categories:
            lbls[img_idx, random_category] = 1

        # Pick random images within those random categories
        random_indices = []
        for i in range(nums_per_image):
            category = random_categories[i]
            random_indices.append(indices_list[category].pop())
            category_nums[category] -= 1

        # Get random images and positions
        mnist_images = []
        positions = []
        for r in random_indices:
            # Get a list containing two PIL images randomly sampled
            # from the database
            img = get_image_from_array(img_data[r])
            img = Image.fromarray(img).resize(
                (original_size, original_size), Image.ANTIALIAS)
            mnist_images.append(img)

            # Generate tuples of (x, y) i.e initial positions for
            # nums_per_image (default: 2)
            positions.append(
                [np.random.rand() * x_lim, np.random.rand() * y_lim])

        positions = np.asarray(positions)

        # Generatte new frames for the entire num_frames
        for frame_idx in range(num_frames):
            canvases = [Image.new("L", (width, height))\
                for _ in range(nums_per_image)]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            # In canv (i.e Image object) place the image at the
            # respective positions
            # Super impose both images on the canvas
            # (i.e empty np array)
            for i, canv in enumerate(canvases):
                canv.paste(
                    mnist_images[i], tuple(positions[i].astype(int)))
                canvas += arr_from_img(canv)

            # Get the next position by adding velocity
            next_pos = positions + veloc

            # Iterate over velocity and see if we hit the wall
            # If we do then change the direction
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        veloc[i] = list(
                            list(veloc[i][:j]) +\
                            [-1 * veloc[i][j]] +\
                            list(veloc[i][j+1:]))

            # Make the permanent change to position by adding
            # updated velocity
            positions = positions + veloc

            # Add the canvas to the dataset array
            imgs[frame_idx, img_idx] =\
                (canvas * 255.).clip(0, 255).astype(np.uint8)

    # shuffle_indices = np.array(range(num_images))
    # np.random.shuffle(shuffle_indices)
    imgs = imgs.transpose(1, 0, 2, 3, 4)
    return imgs, lbls

def main(args):
    img_data, lbl_data = generate_moving_mnist(
        args.mnist_folder,
        args.training,
        shape=(args.frame_size, args.frame_size),
        num_frames=args.num_frames,
        num_images=args.num_images,
        original_size=args.original_size,
        nums_per_image=args.nums_per_image,
        num_single_label=args.num_single_label)

    subfolder = 'train' if args.training else 'val'
    destination_folder = os.path.join(args.dest, subfolder)
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)

    img_folder = os.path.join(destination_folder, 'imgs')
    lbl_folder = os.path.join(destination_folder, 'labels')
    os.makedirs(img_folder)
    os.makedirs(lbl_folder)

    for i in range(len(img_data)):
        img = img_data[i]
        lbl = lbl_data[i]
        np.save(os.path.join(img_folder, '%06d.npy' % i), img)
        np.save(os.path.join(lbl_folder, '%06d.npy' % i), lbl)


if __name__ == "__main__":
    main(args)
