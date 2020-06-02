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

    if find_counter == 0:
        num_gpus = 1
        time = 8
        job.update({
            'name': '2020.06.01',
            'counter': find_counter,
            'config': 'resnet_backbone_points5',
            'criterion': 'backbone_xent', # 'nceprobs_selective',
            'num_routing': 1,
            'dataset': 'shapenet5',
            'batch_size': 72,
            'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results/shapenet',
            'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
            'do_tsne_test_every': 2,
            'do_tsne_test_after': 500, # NOTE: so we aren't doing it rn.
            'lr': 3e-4,
            # 'weight_decay': 5e-4,
            'optimizer': 'adam',
        })

        return find_counter, job
    else:
        print('Counter not found')
        return None, None

if __name__ == '__main__':
    run()
