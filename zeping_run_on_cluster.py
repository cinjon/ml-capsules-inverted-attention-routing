"""Run the jobs in this file.

Example running jobs.

Make a <username>_jobs.py and have it call do_jobarray.
"""
import itertools
import os
import socket
import sys
import getpass

def do_jobarray(email, num_gpus, counter, job, time):
    num_cpus = num_gpus * 4
    gb = num_gpus * 16
    directory = '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zz2332'
    write_func = write_cims
    job['data_root'] = '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/data/MNIST'
    job['affnist_data_root'] = '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/data/affNIST'

    slurm_logs = os.path.join(directory, 'slurm_logs')
    slurm_scripts = os.path.join(directory, 'slurm_scripts')
    if not os.path.exists(slurm_logs):
        os.makedirs(slurm_logs)
    if not os.path.exists(slurm_scripts):
        os.makedirs(slurm_scripts)

    if 'num_gpus' not in job:
        job['num_gpus'] = num_gpus
    job['time'] = time
    job['num_workers'] = min(int(2.5 * num_gpus), num_cpus - num_gpus)
    jobarray = []

    jobname = 'somotion.%dhr.cnt%d' % (time, counter)
    jobcommand = "python train.py --mode jobarray"
    print("Size: ", len(jobarray), jobcommand, " /.../ ", jobname)

    slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')
    write_func(slurmfile, jobname, jobarray, email, num_cpus, time, num_gpus, gb, slurm_logs, jobcommand)

    s = "sbatch %s" % os.path.join(slurm_scripts, jobname + ".slurm")
    os.system(s)

def _write_common(f, jobname, jobarray, email, num_cpus, hours, minutes, num_gpus, gb, slurm_logs):
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --job-name=%s\n" % jobname)
    f.write("#SBATCH --array=%s\n" % ','.join([str(c) for c in jobarray]))
    f.write("#SBATCH --mail-type=END,FAIL\n")
    f.write("#SBATCH --mail-user=%s\n" % email)
    f.write("#SBATCH --cpus-per-task=%d\n" % num_cpus)
    f.write("#SBATCH --time=%d:%d:00\n" % (hours, minutes))
    f.write("#SBATCH --gres=ntasks-per-node=1\n")
    f.write("#SBATCH --mem=%dG\n" % gb)
    f.write("#SBATCH --nodes=%d\n" % 1)
    f.write("#SBATCH --output=%s\n" % os.path.join(
        slurm_logs, jobname + ".%A.%a.out"))
    f.write("#SBATCH --error=%s\n" % os.path.join(
        slurm_logs, jobname + ".%A.%a.err"))


def write_cims(slurmfile, jobname, jobarray, email, num_cpus, time, num_gpus, gb, slurm_logs, code_directory, jobcommand):
    hours = int(time)
    minutes = int((time - hours) * 60)

    # exclude = 'vine[3-14],hpc[1-9],rose[1-4,7-9]'
    exclude = 'vine[3-14],rose[1-4,7-9]'
    with open(slurmfile, 'w') as f:
        _write_common(f, jobname, jobarray, email, num_cpus, hours, minutes, num_gpus, gb, slurm_logs)

        f.write("#SBATCH --gres=gpu:%d\n" % num_gpus)
        f.write("#SBATCH --exclude=%s\n" % exclude)
        # f.write("module purge" + "\n")
        # f.write("module load cuda-10.2\n")
        # f.write("module load gcc-8.1\n")
        # f.write("source activate vidcaps\n")
        # f.write("SRCDIR=%s\n" % code_directory)
        # f.write("cd ${SRCDIR}\n")
        f.write(jobcommand + "\n")
