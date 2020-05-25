"""Run the jobs in this file.

Example running jobs:

python cinjon_point_jobs.py

When you want to add more jobs just put them below and MAKE SURE that all of the
do_jobs for the ones above are False.
"""
from run_on_cluster import do_jobarray
import os
import socket
import getpass
hostname = socket.gethostname()
is_cims = any([
    hostname.startswith('cassio'),
    hostname.startswith('lion'),
    hostname.startswith('weaver')
])
is_dgx = hostname.startswith('dgx-1')
is_prince = hostname.startswith('log-') or hostname.startswith('gpu-')

email = 'cinjon@nyu.edu'
if is_cims or is_dgx:
    code_directory = '/home/resnick/Code/ml-capsules-inverted-attention-routing'
elif is_prince:
    code_directory = '/home/cr2668/Code/ml-capsules-inverted-attention-routing'
else:
    raise


def run(find_counter=None):
    counter = 1

    job = {
        'name': '2020.04.18',
        'optimizer': 'adam',
        'config': 'resnet_backbone_movingmnist2_2cc',
        'criterion': 'triangle',
        'triangle_lambda': 1,
        'num_routing': 1,
        'dataset': 'MovingMNist2',
        'batch_size': 12,
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/mnist'
    }

    num_gpus = 1
    time = 8
    var_arrays = {
        'lr': [.0003],
        'triangle_lambda': [1., 3.],
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job
