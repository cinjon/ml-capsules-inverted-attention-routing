"""Run the jobs in this file.

Example running jobs.

Make a <username>_jobs.py and have it call do_jobarray. 
"""
import itertools
import os
import sys


def do_jobarray(email, code_directory, num_gpus, counter, job, var_arrays,
                time, find_counter, do_job=False):                
    # TODO: Are these numbers optimal?
    num_cpus = num_gpus * 4
    gb = num_gpus * 16
    directory = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps'
    slurm_logs = os.path.join(directory, 'slurm_logs')
    slurm_scripts = os.path.join(directory, 'slurm_scripts')

    job['num_gpus'] = num_gpus
    job['time'] = time
    job['batch_size'] *= num_gpus
    job['num_workers'] = min(int(2.5 * num_gpus), num_cpus - num_gpus)
    jobarray = []

    original_counter = counter

    keys = sorted(var_arrays.keys())
    lsts = [var_arrays[key] for key in keys]
    for job_lst in itertools.product(lsts):
        _job = {k: v for k, v in job.items()}
        _job['counter'] = counter
        for key, value in zip(keys, job_lst):
            _job[key] = value

        if find_counter == counter:
            return counter, _job

        jobarray.append(counter)
        counter += 1

    if find_counter or not do_job:
        return counter, None
                                
    jobname = 'somotion.%dhr.cnt%d-%d' % (time, original_counter, counter)
    jobcommand = "python train.py --mode jobarray"
    print("Size: ", len(jobarray), jobcommand, " /.../ ", jobname)
    
    slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')
    hours = int(time)
    minutes = int((time - hours) * 60)
    with open(slurmfile, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=%s\n" % jobname)
        f.write("#SBATCH --array=%s\n" % ','.join([str(c) for c in jobarray]))
        f.write("#SBATCH --mail-type=END,FAIL\n")
        f.write("#SBATCH --mail-user=%s\n" % email)
        f.write("#SBATCH --cpus-per-task=%d\n" % num_cpus)
        f.write("#SBATCH --time=%d:%d:00\n" % (hours, minutes))
        f.write("#SBATCH --gres=ntasks-per-node=1\n")
        f.write("#SBATCH --gres=gpu:%d\n" % num_gpus)
        f.write("#SBATCH --mem=%dG\n" % gb)
        f.write("#SBATCH --nodes=%d\n" % 1)
        # f.write("#SBATCH --exclude=%s\n" % exclude)
        f.write("#SBATCH --output=%s\n" % os.path.join(
            slurm_logs, jobname + ".%A.%a.out"))
        f.write("#SBATCH --error=%s\n" % os.path.join(
            slurm_logs, jobname + ".%A.%a.err"))
        
        f.write("module purge" + "\n")
        f.write("module load cuda-10.1\n")
        f.write("module load gcc-8.1\n")
        f.write("source activate vidcaps\n")
        f.write("SRCDIR=%s\n" % code_directory)
        f.write("cd ${SRCDIR}\n")
        f.write(jobcommand + "\n")

    s = "sbatch %s" % os.path.join(slurm_scripts, jobname + ".slurm")
    os.system(s)
    return counter, None
