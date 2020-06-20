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
        find_counter=find_counter, do_job=False)
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


    # Counter: 28 ... on Dgx so w more batch size. But also adding in regularization
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
        'weight_decay': 5e-4,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
    }
    num_gpus = 2
    time = 8
    var_arrays = {'lr': [1e-3], 'nce_presence_temperature': [0.1, .03]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 30 ... on Dgx so w more batch size. Regularization and using scheduler 
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
        'weight_decay': 5e-4,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'use_scheduler': True
    }
    num_gpus = 2
    time = 8
    var_arrays = {'lr': [1e-3], 'nce_presence_temperature': [0.1]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 31. Lower LR. Also saving the model so we can run tests afterward.
    job = {
        'name': '2020.06.15',
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
        'weight_decay': 5e-4,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'nce_presence_temperature': 0.1
    }
    num_gpus = 2
    time = 8
    var_arrays = {'lr': [1e-4, 3e-4]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 33. Smaller param config but more conv. Running on CIMS.
    job = {
        'name': '2020.06.15',
        'config': 'resnet_backbone_points16_3conv1fc',
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
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'nce_presence_temperature': 0.1
    }
    num_gpus = 2
    time = 8
    var_arrays = {'lr': [1e-4, 3e-4]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 35. Smaller param config but more conv. Running on CIMS.
    job = {
        'name': '2020.06.15',
        'config': 'resnet_backbone_points16_4conv1fc',
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
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'nce_presence_temperature': 0.1
    }
    num_gpus = 2
    time = 8
    var_arrays = {'lr': [1e-4, 3e-4]}
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 37. The above didn't work. Can we regularize training w rotation?
    # Let's try doing htat.
    job = {
        'name': '2020.06.15',
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
        'shapenet_rotation_train': '-30,30',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'nce_presence_temperature': 0.1,
        'lr': 3e-4
    }
    num_gpus = 2
    time = 8
    var_arrays = {
        'config': ['resnet_backbone_points16_4conv1fc', 'resnet_backbone_points16_3conv1fc']
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 39. Trying -30,30 rotation w lower LR.
    job = {
        'name': '2020.06.15',
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
        'shapenet_rotation_train': '-30,30',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'nce_presence_temperature': 0.1,
        'lr': 1e-4
    }
    num_gpus = 2
    time = 8
    var_arrays = {
        'config': ['resnet_backbone_points16_4conv1fc', 'resnet_backbone_points16_3conv1fc']
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 41. Trying -90,90 rotation.
    job = {
        'name': '2020.06.15',
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
        'shapenet_rotation_train': '-90,90',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'nce_presence_temperature': 0.1,
        'lr': 1e-4
    }
    num_gpus = 2
    time = 8
    var_arrays = {
        'config': ['resnet_backbone_points16_4conv1fc', 'resnet_backbone_points16_3conv1fc']
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 43. Backbone with smaller output_dim.
    job = {
        'name': '2020.06.16',
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
    var_arrays = {
        'config': [
            'resnet_backbone_points16_smbone',
            'resnet_backbone_points16_smbone2',
            'resnet_backbone_points16_smbone3',
            'resnet_backbone_points16_smbone3_gap',
        ]
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 47. Trying with smaller config, has GAP too.
    job = {
        'name': '2020.06.16',
        'config': 'resnet_backbone_points16_smbone3_gap',
        'criterion': 'nceprobs_selective',
        'num_output_classes': 16,
        'num_routing': 1,
        'dataset': 'shapenet16',
        'batch_size': 24,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 5,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'nce_presence_temperature': 0.1,
        'lr': 3e-4
    }
    num_gpus = 2
    time = 8
    var_arrays = {
        'shapenet_rotation_train': ['', '-30,30', '-60,60']
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 50. It's way overfitting on the smaller dataset. So let's add
    # data. Using the _gap models too.
    job = {
        'name': '2020.06.16',
        'config': 'resnet_backbone_points16_smbone3_gap',
        'criterion': 'nceprobs_selective',
        'num_output_classes': 55,
        'num_routing': 1,
        'dataset': 'shapenetFull',
        'batch_size': 24,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 5,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'nce_presence_temperature': 0.1,
        'lr': 3e-4
    }
    num_gpus = 2
    time = 8
    var_arrays = {
        'config': ['resnet_backbone_points16_smbone3_gap',
                   'resnet_backbone_points16_smbone4_gap']
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 52. These should go for longer.  (putting on cims)
    job = {
        'name': '2020.06.16',
        'config': 'resnet_backbone_points16_smbone3_gap',
        'criterion': 'nceprobs_selective',
        'num_output_classes': 55,
        'num_routing': 1,
        'dataset': 'shapenetFull',
        'batch_size': 18,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 5,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'nce_presence_temperature': 0.1,
        'do_modelnet_test_after': 500,
        'lr': 3e-4
    }
    num_gpus = 2
    time = 24
    var_arrays = {
        'config': ['resnet_backbone_points16_smbone3_gap',
                   'resnet_backbone_points16_smbone4_gap'],
        'nce_presence_temperature': [0.1, 0.03]
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 56.
    # These will be doing faster scheduler milestones.
    job = {
        'name': '2020.06.17',
        'config': 'resnet_backbone_points16_smbone3_gap',
        'criterion': 'nceprobs_selective',
        'num_output_classes': 55,
        'num_routing': 1,
        'dataset': 'shapenetFull',
        'batch_size': 18,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 5,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'schedule_milestones': '10,30',
        'nce_presence_temperature': 0.1,
        'lr': 3e-4
    }
    num_gpus = 2
    time = 24
    var_arrays = {
        'config': ['resnet_backbone_points16_smbone3_gap',
                   'resnet_backbone_points16_smbone4_gap'],
        'nce_presence_temperature': [0.1, 0.03]
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 60. Add in the stepsize range change.
    # These will be doing faster scheduler milestones.
    job = {
        'name': '2020.06.18',
        'config': 'resnet_backbone_points16_smbone3_gap',
        'criterion': 'nceprobs_selective',
        'num_output_classes': 55,
        'num_routing': 1,
        'dataset': 'shapenetFull',
        'batch_size': 18,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 5,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '-1,1',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'schedule_milestones': '10,30',
        'nce_presence_temperature': 0.1,
        'lr': 3e-4
    }
    num_gpus = 2
    time = 24
    var_arrays = {
        'config': ['resnet_backbone_points16_smbone3_gap'],
        'nce_presence_temperature': [0.1, 0.03]
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=True)
    if find_counter and _job:
        return counter, _job


    # Counter: 62. These are doign num_routing=2. 
    job = {
        'name': '2020.06.18',
        'config': 'resnet_backbone_points16_smbone3_gap',
        'criterion': 'nceprobs_selective',
        'num_output_classes': 55,
        'num_routing': 2,
        'dataset': 'shapenetFull',
        'batch_size': 12,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 5,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'schedule_milestones': '10,30',
        'nce_presence_temperature': 0.1,
        'lr': 3e-4
    }
    num_gpus = 2
    time = 24
    var_arrays = {
        'config': ['resnet_backbone_points16_smbone3_gap'],
        'nce_presence_temperature': [0.1, 0.03]
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 64:
    # Testing what happens if train replaces test.
    job = {
        'name': '2020.06.20',
        'config': 'resnet_backbone_points16_smbone3_gap',
        'criterion': 'nceprobs_selective',
        'num_output_classes': 55,
        'num_routing': 1,
        'dataset': 'shapenetFull',
        'batch_size': 24,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 5,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 150,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'schedule_milestones': '10,30',
        'nce_presence_temperature': 0.1,
        'lr': 3e-4
    }
    num_gpus = 2
    time = 24
    var_arrays = {
        'config': ['resnet_backbone_points16_smbone3_gap',
                   'resnet_backbone_points16_smbone4_gap'],
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 66. We have test_loader is shuffling. And we are using 5x more
    # caps in the dim. Hugo was saying this might help with the linear on the end.
    job = {
        'name': '2020.06.20',
        'config': 'resnet_backbone_points55_smbone3_gap',
        'criterion': 'nceprobs_selective',
        'num_output_classes': 55,
        'num_routing': 1,
        'dataset': 'shapenetFull',
        'batch_size': 24,
        'optimizer': 'adam',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
        'do_tsne_test_every': 5,
        'do_tsne_test_after': -1,
        'weight_decay': 0,
        'presence_type': 'l2norm',
        'simclr_selection_strategy': 'anchor0_other12',
        'epoch': 350,
        'use_diff_object': True,
        'shapenet_stepsize_range': '0,0',
        'shapenet_rotation_train': '',
        'shapenet_rotation_test': '',
        'use_scheduler': True,
        'schedule_milestones': '10,30',
        'nce_presence_temperature': 0.1,
        'lr': 3e-4
    }
    num_gpus = 2
    time = 24
    var_arrays = {
        'nce_presence_temperature': [0.1, 0.03]
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


if __name__ == '__main__':
    run()
