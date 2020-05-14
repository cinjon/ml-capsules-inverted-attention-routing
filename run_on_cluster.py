"""Run the jobs in this file.

Example running jobs.

Make a <username>_jobs.py and have it call do_jobarray. 
"""
import itertools
import os
import socket
import sys


hostname = socket.gethostname()
is_cims = any([
    hostname.startswith('cassio'),
    hostname.startswith('dgx'),
    hostname.startswith('lion'),
    hostname.startswith('weaver')
])
is_prince = hostname.startswith('log-') or hostname.startswith('gpu-')

def do_jobarray(email, code_directory, num_gpus, counter, job, var_arrays,
                time, find_counter, do_job=False):                
    # TODO: Are these numbers optimal?
    num_cpus = num_gpus * 4
    gb = num_gpus * 16
    if is_cims:
        directory = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps'
        write_func = write_cims
    elif is_prince:
        directory = '/beegfs/cr2668/vidcaps'
        write_func = write_prince
        job['data_root'] = os.path.join(directory, 'MovingMNist')
        job['affnist_data_root'] = os.path.join(directory, 'affnist')
    else:
        raise

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

    original_counter = counter

    keys = sorted(var_arrays.keys())
    lsts = [var_arrays[key] for key in keys]
    for job_lst in itertools.product(*lsts):
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
    write_func(slurmfile, jobname, jobarray, email, num_cpus, time, num_gpus, gb, slurm_logs, code_directory, jobcommand)

    s = "sbatch %s" % os.path.join(slurm_scripts, jobname + ".slurm")
    os.system(s)
    return counter, None


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

    exclude = 'vine[3-14],hpc[1-9],rose[1-4,7-9]'
    with open(slurmfile, 'w') as f:
        _write_common(f, jobname, jobarray, email, num_cpus, hours, minutes, num_gpus, gb, slurm_logs)

        f.write("#SBATCH --gres=gpu:%d\n" % num_gpus)
        f.write("#SBATCH --exclude=%s\n" % exclude)        
        f.write("module purge" + "\n")
        f.write("module load cuda-10.2\n")
        f.write("module load gcc-8.1\n")
        f.write("source activate vidcaps\n")
        f.write("SRCDIR=%s\n" % code_directory)
        f.write("cd ${SRCDIR}\n")
        f.write(jobcommand + "\n")


def write_prince(slurmfile, jobname, jobarray, email, num_cpus, time, num_gpus, gb, slurm_logs, code_directory, jobcommand):
    hours = int(time)
    minutes = int((time - hours) * 60)

    with open(slurmfile, 'w') as f:
        _write_common(f, jobname, jobarray, email, num_cpus, hours, minutes, num_gpus, gb, slurm_logs)

        f.write("#SBATCH --gres=gpu:p40:%d\n" % num_gpus)
        # f.write("#SBATCH --partition=p40\n")
        f.write("module purge\n")
        f.write("module load cudnn/10.1v7.6.5.32\n")
        f.write("module load cuda/10.1.105\n")
        f.write("source /beegfs/cr2668/venvs/vidcaps/bin/activate\n")
        f.write("SRCDIR=%s\n" % code_directory)
        f.write("cd ${SRCDIR}\n")
        f.write(jobcommand + "\n")
