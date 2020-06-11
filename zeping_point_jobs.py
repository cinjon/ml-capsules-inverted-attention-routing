"""Run the jobs in this file.

Example running jobs:

python zeping_jobs.py

When you want to add more jobs just put them below and MAKE SURE that all of the
do_jobs for the ones above are False.
"""
from zeping_run_on_cluster import do_jobarray

email = 'zz2332@nyu.edu'
code_directory = '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/ml-capsules-inverted-attention-routing'

def run(find_counter=None):
    job = {
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/data/MNIST',
        'affnist_data_root': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/data/affNIST'
    }

    # 0: resnet backbone xent
    if find_counter == 0:
        num_gpus = 1
        time = 8
        job.update({
            'name': '2020.06.08',
            'counter': find_counter,
            'config': 'resnet_backbone_points5',
            'criterion': 'backbone_xent', # 'nceprobs_selective',
            'num_routing': 1,
            'dataset': 'shapenet5',
            'batch_size': 16,
            'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results/shapenet',
            'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
            'do_tsne_test_every': 2,
            'do_tsne_test_after': 500, # NOTE: so we aren't doing it rn.
            'lr': 3e-4,
            # 'weight_decay': 5e-4,
            'optimizer': 'adam',
            'epochs': 200
        })
        return find_counter, job

    # 1-2: pointcapsnet 16 backbone xent
    # vars: weight_decay
    if find_counter in [1, 2]:
        num_gpus = 1
        time = 8
        job.update({
            'name': '2020.06.08',
            'counter': find_counter,
            'config': 'pointcapsnet_backbone_points5_cap16',
            'criterion': 'backbone_xent', # 'nceprobs_selective',
            'num_routing': 1,
            'dataset': 'shapenet5',
            'batch_size': 16,
            'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results/shapenet',
            'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
            'do_tsne_test_every': 2,
            'do_tsne_test_after': 500, # NOTE: so we aren't doing it rn.
            'lr': 3e-4,
            'optimizer': 'adam',
            'epoch': 200
        })

        if find_counter == 1:
            job.update({'weight_decay': 0})
        if find_counter == 2:
            job.update({'weight_decay': 5e-4})

        return find_counter, job

    # 3-4: pointcapsnet 8 backbone xent
    # vars: weight_decay
    if find_counter in [3, 4]:
        num_gpus = 1
        time = 8
        job.update({
            'name': '2020.06.08',
            'counter': find_counter,
            'config': 'pointcapsnet_backbone_points5_cap8',
            'criterion': 'backbone_xent', # 'nceprobs_selective',
            'num_routing': 1,
            'dataset': 'shapenet5',
            'batch_size': 16,
            'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results/shapenet',
            'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
            'do_tsne_test_every': 2,
            'do_tsne_test_after': 500, # NOTE: so we aren't doing it rn.
            'lr': 3e-4,
            'optimizer': 'adam',
            'epoch': 200
        })

        if find_counter == 3:
            job.update({
                'weight_decay': 0
            })
        if find_counter == 4:
            job.update({
                'weight_decay': 5e-4
            })

        return find_counter, job

    # 5-8: pointcapsnet 8 backbone nce
    # vars: weight_decay, nce_presence_temperature
    if find_counter in [5, 6, 7, 8]:
        num_gpus = 1
        time = 8
        job.update({
            'name': '2020.06.08',
            'counter': find_counter,
            'config': 'pointcapsnet_backbone_points5_cap8',
            'criterion': 'backbone_nceprobs_selective',
            'num_routing': 1,
            'dataset': 'shapenet5',
            'batch_size': 16,
            'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results/shapenet',
            'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
            'do_tsne_test_every': 2,
            'do_tsne_test_after': -1, # 500,
            'lr': 3e-4,
            'optimizer': 'adam',
            'presence_type': 'l2norm',
            'simclr_selection_strategy': 'anchor0_other12',
            'nce_presence_lambda': 1.0,
            'epoch': 200
        })

        if find_counter == 5:
            job.update({
                'weight_decay': 0,
                'nce_presence_temperature': 0.1
            })
        if find_counter == 6:
            job.update({
                'weight_decay': 0,
                'nce_presence_temperature': 0.01
            })
        if find_counter == 7:
            job.update({
                'weight_decay': 5e-4,
                'nce_presence_temperature': 0.1
            })
        if find_counter == 8:
            job.update({
                'weight_decay': 5e-4,
                'nce_presence_temperature': 0.01
            })

        return find_counter, job

    # 9-12: pointcapsnet 16 backbone nce
    # vars: weight_decay, nce_presence_temperature
    if find_counter in [9, 10, 11, 12]:
        num_gpus = 1
        time = 8
        job.update({
            'name': '2020.06.08',
            'counter': find_counter,
            'config': 'pointcapsnet_backbone_points5_cap16',
            'criterion': 'backbone_nceprobs_selective',
            'num_routing': 1,
            'dataset': 'shapenet5',
            'batch_size': 16,
            'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results/shapenet',
            'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
            'do_tsne_test_every': 2,
            'do_tsne_test_after': -1, # 500,
            'lr': 3e-4,
            'optimizer': 'adam',
            'presence_type': 'l2norm',
            'simclr_selection_strategy': 'anchor0_other12',
            'nce_presence_lambda': 1.0,
            'epoch': 200
        })

        if find_counter == 9:
            job.update({
                'weight_decay': 0,
                'nce_presence_temperature': 0.1
            })
        if find_counter == 10:
            job.update({
                'weight_decay': 0,
                'nce_presence_temperature': 0.01
            })
        if find_counter == 11:
            job.update({
                'weight_decay': 5e-4,
                'nce_presence_temperature': 0.1
            })
        if find_counter == 12:
            job.update({
                'weight_decay': 5e-4,
                'nce_presence_temperature': 0.01
            })

        return find_counter, job

    # 13-16: pointcapsnet cap8 backbone nce with different object and same origin
    # vars: weight_decay, nce_presence_temperature
    if find_counter in [13, 14, 15, 16]:
        num_gpus = 1
        time = 8
        job.update({
            'name': '2020.06.08',
            'counter': find_counter,
            'config': 'pointcapsnet_backbone_points5_cap8',
            'criterion': 'backbone_nceprobs_selective',
            'num_routing': 1,
            'dataset': 'shapenet5',
            'batch_size': 16,
            'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results/shapenet',
            'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
            'do_tsne_test_every': 2,
            'do_tsne_test_after': -1, # 500,
            'lr': 3e-4,
            'optimizer': 'adam',
            'presence_type': 'l2norm',
            'simclr_selection_strategy': 'anchor0_other12',
            'nce_presence_lambda': 1.0,
            '--shapenet_stepsize_range': '0,0',
            '--shapenet_rotation_train': '',
            '--shapenet_rotation_test': '',
            'use_diff_object': True,
            'epoch': 200
        })

        if find_counter == 13:
            job.update({
                'weight_decay': 0,
                'nce_presence_temperature': 0.1
            })
        if find_counter == 14:
            job.update({
                'weight_decay': 0,
                'nce_presence_temperature': 0.01
            })
        if find_counter == 15:
            job.update({
                'weight_decay': 5e-4,
                'nce_presence_temperature': 0.1
            })
        if find_counter == 16:
            job.update({
                'weight_decay': 5e-4,
                'nce_presence_temperature': 0.01
            })

        return find_counter, job

    # 17-20: pointcapsnet cap16 backbone nce with different object,
    # same origin and no rotation
    # vars: weight_decay, nce_presence_temperature
    if find_counter in [17, 18, 19, 20]:
        num_gpus = 1
        time = 8
        job.update({
            'name': '2020.06.08',
            'counter': find_counter,
            'config': 'pointcapsnet_backbone_points5_cap16',
            'criterion': 'backbone_nceprobs_selective',
            'num_routing': 1,
            'dataset': 'shapenet5',
            'batch_size': 16,
            'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results/shapenet',
            'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
            'do_tsne_test_every': 2,
            'do_tsne_test_after': -1, # 500,
            'lr': 3e-4,
            'optimizer': 'adam',
            'presence_type': 'l2norm',
            'simclr_selection_strategy': 'anchor0_other12',
            'nce_presence_lambda': 1.0,
            '--shapenet_stepsize_range': '0,0',
            '--shapenet_rotation_train': '',
            '--shapenet_rotation_test': '',
            'use_diff_object': True,
            'epoch': 200
        })

        if find_counter == 17:
            job.update({
                'weight_decay': 0,
                'nce_presence_temperature': 0.1
            })
        if find_counter == 18:
            job.update({
                'weight_decay': 0,
                'nce_presence_temperature': 0.01
            })
        if find_counter == 19:
            job.update({
                'weight_decay': 5e-4,
                'nce_presence_temperature': 0.1
            })
        if find_counter == 20:
            job.update({
                'weight_decay': 5e-4,
                'nce_presence_temperature': 0.01
            })

        return find_counter, job

    # 21-22: pointcapsnet backbone nce on dataset16
    # vars: nce_presence_temperature
    if find_counter in [21, 22]:
        num_gpus = 1
        time = 8
        job.update({
            'name': '2020.06.10',
            'counter': find_counter,
            'config': 'pointcapsnet_backbone_points5_cap16',
            'criterion': 'backbone_nceprobs_selective',
            'num_routing': 1,
            'dataset': 'shapenet16',
            'batch_size': 16,
            'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results/shapenet',
            # 'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
            'data_root': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/shapenet',
            'do_tsne_test_every': 2,
            'do_tsne_test_after': -1,
            'lr': 3e-4,
            'optimizer': 'adam',
            'presence_type': 'l2norm',
            'simclr_selection_strategy': 'anchor0_other12',
            'nce_presence_lambda': 1.0,
            # '--shapenet_stepsize_range': '0,0',
            # '--shapenet_rotation_train': '',
            # '--shapenet_rotation_test': '',
            # 'use_diff_object': True,
            'epoch': 350,
            'weight_decay': 0,
        })

        if find_counter == 21:
            job.update({
                'nce_presence_temperature': 0.1
            })
        if find_counter == 22:
            job.update({
                'nce_presence_temperature': 0.01
            })

        return find_counter, job

    # 23-24: pointcapsnet backbone nce on dataset16 with different object,
    # same origin and no rotation
    # vars: nce_presence_temperature
    if find_counter in [23, 24]:
        num_gpus = 1
        time = 8
        job.update({
            'name': '2020.06.10',
            'counter': find_counter,
            'config': 'pointcapsnet_backbone_points5_cap16',
            'criterion': 'backbone_nceprobs_selective',
            'num_routing': 1,
            'dataset': 'shapenet16',
            'batch_size': 16,
            'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results/shapenet',
            # 'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
            'data_root': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/shapenet',
            'do_tsne_test_every': 2,
            'do_tsne_test_after': -1, # 500,
            'lr': 3e-4,
            'optimizer': 'adam',
            'presence_type': 'l2norm',
            'simclr_selection_strategy': 'anchor0_other12',
            'nce_presence_lambda': 1.0,
            '--shapenet_stepsize_range': '0,0',
            '--shapenet_rotation_train': '',
            '--shapenet_rotation_test': '',
            'use_diff_object': True,
            'epoch': 350,
            'weight_decay': 0,
        })

        if find_counter == 23:
            job.update({
                'nce_presence_temperature': 0.1
            })
        if find_counter == 24:
            job.update({
                'nce_presence_temperature': 0.01
            })

        return find_counter, job

    else:
        print('Counter not found')
        return None, None

if __name__ == '__main__':
    run()
