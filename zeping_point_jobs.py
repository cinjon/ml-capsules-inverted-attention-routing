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



if __name__ == '__main__':
    run()
