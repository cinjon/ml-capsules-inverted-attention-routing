"""Run the jobs in this file.

Example running jobs:

python zeping_jobs.py

When you want to add more jobs just put them below and MAKE SURE that all of the
do_jobs for the ones above are False.
"""
from run_on_cluster import do_jobarray

email = 'zz2332@nyu.edu'
# TODO: Put in correct code_directory!
code_directory = '/home/resnick/Code/ml-capsules-inverted-attention-routing'


def run(find_counter=None):
    counter = 1

    # NOTE: See example in cinjn_jobs.py
    job = {} 
    var_arrays = {}
    num_gpus = 2
    time = 24
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=True)
    if find_counter and _job:
        return counter, _job
    return None, None
            
                                
if __name__ == '__main__':
    run()
