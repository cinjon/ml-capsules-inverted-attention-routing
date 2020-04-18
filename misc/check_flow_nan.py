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

def f(path):
    print(path, flush=True)
    bad_flows = []
    flow_paths = glob(os.path.join(path, "*.npy"))
    for flow_path in flow_paths:
        flow = np.load(flow_path)
        if np.sum(np.isnan(flow)) != 0:
            if len(bad_flows) == 0:
                bad_flows.append(os.path.basename(path[:-1]))
            bad_flows.append(os.path.basename(flow_path))
    return bad_flows

if __name__ == "__main__":
    p = Pool(30)
    bad_list = p.map(f, sorted(glob(os.path.join(args.root, "*/"))))
    bad_dict = {}
    for item in bad_list:
        if item == []:
            continue
        bad_dict[item[0]] = item[1:]
    with open(os.path.join(args.store_dir, 'nan_flow_list.json'), 'w') as f:
        json.dump(bad_dict, f)
