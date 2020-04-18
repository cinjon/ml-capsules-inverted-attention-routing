"""Run the jobs in this file.

Example running jobs:

python cinjon_jobs.py

When you want to add more jobs just put them below and MAKE SURE that all of the
do_jobs for the ones above are False.
"""
from run_on_cluster import do_jobarray

email = 'cinjon@nyu.edu'
code_directory = '/home/resnick/Code/ml-capsules-inverted-attention-routing'


def run(find_counter=None):
    counter = 1

    # NOTE: These jobs are running the triangle loss with MovingMnist2.
    # They are using two capsules, so we may run into issues with the
    # capsules not being ordered correctly.
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
        find_counter=find_counter, do_job=True)
    if find_counter and _job:
        return counter, _job

    # NOTE: These jobs are running the triangle loss in order to get a sense of
    # if that trains. It's working w Gymnastics but only using one capsule at
    # the end.
    job = {
        'name': '2020.04.18',
        'optimizer': 'adam',
        'config': 'resnet_backbone_gymnastics_1cc',
        'criterion': 'triangle',
        'triangle_lambda': 1,
        'num_routing': 1,
        'dataset': 'gymnasticsRgbFrame',
        'skip_videoframes': 6,
        'dist_videoframes': -19,
        'num_videoframes': 5,
        'batch_size': 4,
        'video_names': '2628',
        'gymnastics_video_directory': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/frames.feb102020',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results',
    }
    num_gpus = 1
    time = 8
    var_arrays = {
        'lr': [.0003],
        'triangle_lambda': [1., 3, 10.],
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=True)
    if find_counter and _job:
        return counter, _job

    # NOTE: These jobs are running the triangle loss in order to get a sense of
    # if that trains. It's working w Gymnastics but only using one capsule at
    # the end.
    job = {
        'name': '2020.04.18',
        'optimizer': 'adam',
        'config': 'resnet_backbone_gymnastics_1cc',
        'criterion': 'triangle',
        'triangle_lambda': 1,
        'num_routing': 1,
        'dataset': 'gymnasticsRgbFrame',
        'skip_videoframes': 6,
        'dist_videoframes': -19,
        'num_videoframes': 5,
        'batch_size': 4,
        'video_names': '2628,2348,2654,2846,2660,2843',
        'gymnastics_video_directory': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/frames.feb102020',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results',
    }
    num_gpus = 1
    time = 8
    var_arrays = {
        'lr': [.0003],
        'triangle_lambda': [1., 3, 10.],
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    return None, None
            
                                
if __name__ == '__main__':
    run()
