import argparse
import os
import json
import sys

import cv2
import numpy as np


def create_flow2img_dir(phase_dir=None, subdir=None, output_dir=None, input_dir=None):
    """
    Args:
      phase_dir: The directory of directory of flow npys.
      subdir: The particular subdir within that phase_dir
      output_dir: The directory to output the rgb representations.

    Returns:
      the number of files that we created.
    """
    if input_dir is None:
        input_dir = os.path.join(phase_dir, subdir)
        target_dir = os.path.join(output_dir, subdir)
    else:
        target_dir = output_dir

    files = [f for f in os.listdir(input_dir) if f.endswith('npy')]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for f in files:
        flow = np.load(os.path.join(input_dir, f))
        hsv = np.zeros([flow.shape[0], flow.shape[1], 3])
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,1] = 255
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        np.save(os.path.join(target_dir, f), rgb)
    return len(files)


def create_flow2img_phase(phase_dir, output_dir):
    """Create the rgb images for each of the flow npys.

    Args:
      phase_dir: The directory of directory of flow npys. 
      output_dir: The directory to output the rgb representations.
    """
    total = 0
    num = 0
    subdirs = sorted(os.listdir(phase_dir))
    start_time = datetime.datetime.now()
    for num_subdir, subdir in enumerate(subdirs):
        if num_subdir > 0:
            now = datetime.datetime.now()
            ffs = 1.0 * total / (now - start_time).total_seconds()
            print('Frames / Second: %.4f' % ffs)

        print('Doing %s (%d / %d).' % (subdir, num_subdir+1, len(subdirs)))
        file_count = create_flow2img_dir(phase_dir, subdir, output_dir)
        total += file_count

    print('Completed %d / %d.' % (num, total))


def batch_flow2img(args):
    data_root = args.data_root
    slurm_logs = os.path.join(data_root, "slurm_logs")
    slurm_scripts = os.path.join(data_root, "slurm_scripts")
    if not os.path.exists(slurm_logs):
        os.makedirs(slurm_logs)
    if not os.path.exists(slurm_scripts):
        os.makedirs(slurm_scripts)

    phase_dir = os.path.join(data_root, 'flows', args.phase)
    output_dir = os.path.join(data_root, args.output_dir, args.phase)
    subdirs = sorted(os.listdir(phase_dir))
    print('find {} videos.'.format(len(subdirs)))

    for num, subdir in enumerate(subdirs):
        jobname = 'flow2img.n%d-%s' % (num, subdir)
        jobcommand = "python flow2img.py --data_root %s --mode single " \
                     "--phase %s --output_dir %s --subdir %s" % (
                         data_root, args.phase, output_dir, subdir)
                         
        print(jobcommand, " /.../ ", jobname)
        slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')
        with open(slurmfile, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --job-name=%s\n" % jobname)
            # f.write("#SBATCH --qos=batch\n")
            f.write("#SBATCH --mail-type=END,FAIL\n")
            f.write("#SBATCH --mail-user=cinjon@nyu.edu\n")
            f.write("#SBATCH --cpus-per-task=1\n")
            f.write("#SBATCH --time=8:00:00\n") # NOTE: 8hrs.
            f.write("#SBATCH --mem=%dG\n" % 4)
            f.write("#SBATCH --nodes=%d\n" % 1)
            f.write("#SBATCH --output=%s\n" % os.path.join(
                slurm_logs, jobname + ".out"))
            f.write("#SBATCH --error=%s\n" % os.path.join(
                slurm_logs, jobname + ".err"))

            f.write("module purge" + "\n")
            f.write("source activate somtorch\n")
            f.write("SRCDIR=${HOME}/Code/spaceofmotion-ml/munge\n")
            f.write("cd ${SRCDIR}\n")
            f.write(jobcommand + "\n")

        s = "sbatch %s" % os.path.join(slurm_scripts, jobname + ".slurm")
        os.system(s)


def batch_flow2img_dirs(args):
    data_root = args.data_root # /.../sep052019
    slurm_logs = os.path.join(data_root, "slurm_logs")
    slurm_scripts = os.path.join(data_root, "slurm_scripts")
    if not os.path.exists(slurm_logs):
        os.makedirs(slurm_logs)
    if not os.path.exists(slurm_scripts):
        os.makedirs(slurm_scripts)

    input_dir = os.path.join(data_root, args.input_dir)
    output_dir = os.path.join(data_root, args.output_dir)
    subdirs = sorted(os.listdir(input_dir))
    print('find {} videos.'.format(len(subdirs)))

    for num, subdir in enumerate(subdirs):
        if num >= args.e_ or num < args.s_:
            continue

        jobname = 'flow2img.n%d-%s' % (num, subdir)
        jobname = jobname.replace('&#039;', '').replace('&amp;', '').replace('(', '').replace(')', '')
        output_dir_ = os.path.join(output_dir, subdir)
        input_dir_ = os.path.join(input_dir, subdir)
        jobcommand = "python flow2img.py --data_root %s --mode singledir " \
                     '--output_dir "%s" --input_dir "%s"' % (
                         data_root, output_dir_, input_dir_)
                         
        print(jobcommand, " /.../ ", jobname)
        slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')
        with open(slurmfile, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --job-name=%s\n" % jobname)
            # f.write("#SBATCH --qos=batch\n")
            f.write("#SBATCH --mail-type=END,FAIL\n")
            f.write("#SBATCH --mail-user=cinjon@nyu.edu\n")
            f.write("#SBATCH --cpus-per-task=1\n")
            f.write("#SBATCH --time=8:00:00\n") # NOTE: 8hrs.
            f.write("#SBATCH --mem=%dG\n" % 4)
            f.write("#SBATCH --nodes=%d\n" % 1)
            f.write("#SBATCH --output=%s\n" % os.path.join(
                slurm_logs, jobname + ".out"))
            f.write("#SBATCH --error=%s\n" % os.path.join(
                slurm_logs, jobname + ".err"))

            f.write("module purge" + "\n")
            f.write("source activate somtorch\n")
            f.write("SRCDIR=${HOME}/Code/spaceofmotion-ml/munge\n")
            f.write("cd ${SRCDIR}\n")
            f.write(jobcommand + "\n")

        s = "sbatch %s" % os.path.join(slurm_scripts, jobname + ".slurm")
        os.system(s)


def parse_args():
    parser = argparse.ArgumentParser(
        description="convert the flow 2d images into 3d images.")
    parser.add_argument(
        '--data_root',
        default='/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/mar-31-2019',
        type=str)
    parser.add_argument('--phase', default='', type=str)
    parser.add_argument('--output_dir', default='flowimgs', type=str)
    parser.add_argument('--input_dir', default='flowimgs', type=str)
    parser.add_argument('--subdir', default='none', type=str)
    parser.add_argument('--mode', default='none', type=str)
    parser.add_argument('--s_', default=0, type=int)
    parser.add_argument('--e_', default=1, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.mode == "batch":
        batch_flow2img(args)
    elif args.mode == "batchdir":
        batch_flow2img_dirs(args)
    elif args.mode == "single":
        phase_dir = os.path.join(args.data_root, 'flows', args.phase)
        create_flow2img_dir(phase_dir, args.subdir, args.output_dir)
    elif args.mode == "singledir":
        # NOTE: This wants the dirs to be absolute paths.
        create_flow2img_dir(input_dir=args.input_dir, output_dir=args.output_dir)
