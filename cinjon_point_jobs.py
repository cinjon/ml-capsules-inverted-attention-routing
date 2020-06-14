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
if is_prince:
    code_directory = '/home/cr2668/Code/ml-capsules-inverted-attention-routing'
else:
    code_directory = '/home/resnick/Code/ml-capsules-inverted-attention-routing'


def run(find_counter=None):
    counter = 1

    job = {
        'name': '2020.05.27',
        'config': 'resnet_backbone_points5',
        'criterion': 'backbone_xent', # 'nceprobs_selective',
        'num_routing': 1,
        'dataset': 'shapenet5',
        'batch_size': 72,
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 2,
        'do_tsne_test_after': 500, # NOTE: so we aren't doing it rn.
    }
    num_gpus = 1
    time = 8
    var_arrays = {}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Counter: 2. Param count 2417424 because it has linear.
    job = {
        'name': '2020.06.11',
        'config': 'pointcapsnet_backbone_points5_cap16',
        'criterion': 'backbone_xent',
        'num_routing': 1,
        'dataset': 'shapenet16',
        'batch_size': 36,
        'lr': 1e-3,
        'num_output_classes': 16,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 2,
        'do_tsne_test_after': 500, # NOTE: so we aren't doing it rn.
    }
    num_gpus = 1
    time = 5
    var_arrays = {'lr': [3e-3, 1e-3], 'weight_decay': [0, 5e-4]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Counter: 6. Params=2155264
    # These are doing fixed center and no rotation.
    job = {
        'name': '2020.06.11',
        'config': 'pointcapsnet_backbone_points5_cap16',
        'criterion': 'backbone_nceprobs_selective',
        'num_output_classes': 16,
        'num_routing': 1,
        'dataset': 'shapenet16',
        'batch_size': 16,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 2,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'nce_presence_lambda': 1.0,
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': ''
    }
    num_gpus = 2
    time = 10
    var_arrays = {'lr': [3e-4, 1e-3], 'nce_presence_temperature': [0.1, 0.03]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 10. Same as above but with further decrease in presence temp
    # These are doing fixed center and no rotation.
    # Here we aren't doing the linear out, so we have 2155264 params.
    job = {
        'name': '2020.06.11',
        'config': 'pointcapsnet_backbone_points5_cap16',
        'criterion': 'backbone_nceprobs_selective',
        'num_output_classes': 16,
        'num_routing': 1,
        'dataset': 'shapenet16',
        'batch_size': 16,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 2,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'nce_presence_lambda': 1.0,
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': ''
    }
    num_gpus = 2
    time = 10
    var_arrays = {'lr': [3e-4, 1e-3], 'nce_presence_temperature': [0.01, 0.003]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 14. Try the other backbone. What is their comparative sizes?
    # These are doing fixed center and no rotation. Uhhh but then this has
    # a lot of params: 4353360. Yeah ok, it ends with 262144 features, then
    # goes into 16 out_features. That's why it's so big yo.
    job = {
        'name': '2020.06.12',
        'config': 'resnet_backbone_points5',
        'criterion': 'backbone_xent', # 'nceprobs_selective',
        'num_output_classes': 16,
        'num_routing': 1,
        'dataset': 'shapenet16',
        'batch_size': 36,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'weight_decay': 0,
        'do_tsne_test_every': 2,
        'do_tsne_test_after': 500,
        'epoch': 350,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'lr': 1e-3
    }
    num_gpus = 1
    time = 5
    var_arrays = {'lr': [1e-3, 3e-4]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 16. Try the other backbone. What is their comparative sizes?
    # These are doing fixed center and no rotation.
    # When doing this on NCE, it's 159040, which is an oder of magnitude smaller.
    job = {
        'name': '2020.06.12',
        'config': 'resnet_backbone_points5',
        'criterion': 'backbone_nceprobs_selective',
        'num_output_classes': 16,
        'num_routing': 1,
        'dataset': 'shapenet16',
        'batch_size': 16,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 2,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': ''
    }
    num_gpus = 2
    time = 5
    var_arrays = {'lr': [3e-4, 1e-3], 'nce_presence_temperature': [0.1, 0.01]}                  
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 20 ... has 5523112 params.
    # Do the full config, not just the backbone. Not using ModelNet here. Doing
    # fixed center and no rotation.
    job = {
        'name': '2020.06.14',
        'config': 'resnet_backbone_points16',
        'criterion': 'nceprobs_selective',
        'num_output_classes': 16,
        'num_routing': 1,
        'dataset': 'shapenet16',
        'batch_size': 16,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 2,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': ''
    }
    num_gpus = 2
    time = 8
    var_arrays = {'lr': [1e-3], 'nce_presence_temperature': [0.3, 0.1, 0.03]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=True)
    if find_counter and _job:
        return counter, _job


    # Counter: 23 (2256208). We updated the backbone to have stride=2. Ensure
    # this still works to a reasonable extent for XEnt.
    job = {
        'name': '2020.06.14',
        'config': 'resnet_backbone_points16',
        'criterion': 'backbone_xent', # nceprobs_selective',
        'num_output_classes': 16,
        'num_routing': 1,
        'dataset': 'shapenet16',
        'batch_size': 36,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'weight_decay': 0,
        'do_tsne_test_every': 2,
        'do_tsne_test_after': 500,
        'epoch': 350,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'lr': 1e-3
    }
    num_gpus = 1
    time = 5
    var_arrays = {'lr': [1e-3, 3e-4]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 25 ... on Dgx so w more batch size.
    job = {
        'name': '2020.06.14',
        'config': 'resnet_backbone_points16',
        'criterion': 'nceprobs_selective',
        'num_output_classes': 16,
        'num_routing': 1,
        'dataset': 'shapenet16',
        'batch_size': 24,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 2,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': ''
    }
    num_gpus = 2
    time = 8
    var_arrays = {'lr': [1e-3], 'nce_presence_temperature': [0.3, 0.1, 0.03]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


if __name__ == '__main__':
    run()
