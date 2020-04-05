import os
import json
import argparse
import numpy as np
from glob import glob
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    default="/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/flows",
    type=str)
parser.add_argument(
    "--store_dir",
    default="/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion",
    type=str)
args = parser.parse_args()

def f(folder_path):
    print(os.path.basename(folder_path[:-1]))
    magnitude_list = []
    paths = sorted(glob(os.path.join(folder_path, "*.npy")))
    for path in paths:
        flow = np.load(path)
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)

        rad = np.sqrt(np.square(u) + np.square(v))
        magnitude_list.append(np.average(rad))
    return [os.path.basename(folder_path[:-1]), np.array(magnitude_list)]

if __name__ == "__main__":
    p = Pool(30)
    magnitude = p.map(f, sorted(glob(os.path.join(args.root, "*/"))))
    np.save(os.path.join(args.store_dir, "magnitude.npy"), magnitude)
