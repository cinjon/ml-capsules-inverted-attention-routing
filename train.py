"""
NOTE: This is just here to build in linpred_train while other models are trying
to launch.
"""

import os
import getpass
import json
import time
import pickle
import random
import argparse
import numpy as np
from datetime import datetime

from comet_ml import Experiment as CometExperiment, OfflineExperiment
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import configs
import linpred_train
import cinjon_jobs
import zeping_jobs
from src.moving_mnist.moving_mnist import MovingMNist
from src.moving_mnist.moving_mnist2 import MovingMNIST as MovingMNist2
from src.moving_mnist.moving_mnist2 import MovingMNISTClassSampler
from src import capsule_time_model
from src.affnist import AffNist
from src.gymnastics import GymnasticsVideo
from src.gymnastics import GymnasticsRgbFrame
from src.gymnastics.gymnastics_flow import GymnasticsFlow, GymnasticsFlowExperiment, gymnastics_flow_collate
import video_transforms
from src.resnet_reorder_model import ReorderResNet


def run_tsne(data_root, model, path, epoch, use_moving_mnist=False,
             use_mnist=False, comet_exp=None, center_start=True,
             single_angle=False, use_cuda_tsne=False, tsne_batch_size=8,
             num_workers=2):
    # from MulticoreTSNE import MulticoreTSNE as multiTSNE

    # NOTE: Use this when it's the only thing on the GPU.
    if use_cuda_tsne:
        from tsnecuda import TSNE as cudaTSNE
    else:
        from sklearn.manifold import TSNE

    if use_moving_mnist:
        train_set = MovingMNist2(data_root, train=True, seq_len=1,
                                 image_size=64, colored=False, tiny=False,
                                 num_digits=1, one_data_loop=True,
                                 center_start=center_start,
                                 single_angle=single_angle)
        test_set = MovingMNist2(data_root, train=False, seq_len=1,
                                image_size=64, colored=False, tiny=False,
                                num_digits=1, one_data_loop=True,
                                center_start=center_start,
                                single_angle=single_angle)
        suffix = 'movmnist64.pfix%d.afix%d' % (int(center_start), int(single_angle))
    elif use_mnist:
        train_set = torchvision.datasets.MNIST(
            '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/mnist', train=True,
            download=True,
            transform=transforms.Compose([
                # transforms.Resize(64),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        test_set = torchvision.datasets.MNIST(
            '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/mnist', train=False,
            transform=transforms.Compose([
                transforms.Pad(18),
                transforms.RandomAffine(0, (.02, .02)),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        suffix = 'mnistpad18translateRand1'

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=tsne_batch_size,
                                               shuffle=False,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=tsne_batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)

    orig_images = []
    model_poses = []
    model_orig_poses = []
    model_presence = []
    targets = []
    presence_class_probs = None

    with torch.no_grad():
        for loader, split in zip([train_loader, test_loader], ['train', 'test']):
            if split == 'train':
                continue

            for batch_num, (images, labels) in enumerate(loader):
                if batch_num % 10 == 0:
                    print('Batch %d / %d' % (batch_num, len(loader)))

                # if batch_num == 0:
                #     imgs = images[0].squeeze().cpu().numpy()
                #     imgs = (imgs * 255).astype(np.uint8)
                #     imgs = Image.fromarray(imgs)
                #     path_ = os.path.join(path, 'images.%s.png' % suffix)
                #     imgs.save(path_)

                images = images.to('cuda')
                batch_size, num_images = images.shape[:2]

                # Change view so that we can put everything through the model at once.
                images = images.view(batch_size * num_images, *images.shape[2:])
                poses = model(images)
                if type(poses) == tuple:
                    if len(poses) == 2:
                        poses, presence = poses
                        poses = poses.view(batch_size, num_images, *poses.shape[1:])
                        poses = poses[:, 0]
                        presence = presence.view(batch_size, num_images, *presence.shape[1:])
                        presence = presence[:, 0]
                    elif len(poses) == 3:
                        poses, presence, presence_class_probs = poses
                        poses = poses.view(batch_size, num_images, *poses.shape[1:])
                        poses = poses[:, 0]
                        presence = presence.view(batch_size, num_images, *presence.shape[1:])
                        presence = presence[:, 0]
                        presence_class_probs = presence_class_probs.view(batch_size, num_images, *presence_class_probs.shape[1:])
                        presence_class_probs = presence_class_probs[:, 0]
                    else:
                        raise
                    orig_poses = poses.cpu()
                    poses *= presence[:, :, None]
                    presence = presence.cpu()
                    model_orig_poses.append(orig_poses.view(batch_size, -1))
                    model_presence.append(presence.view(batch_size, -1))
                else:
                    poses = poses.view(batch_size, num_images, *poses.shape[1:])
                    poses = poses[:, 0]

                poses = poses.cpu()
                images = images.cpu()
                del images
                model_poses.append(poses.view(batch_size, -1))
                targets.append(labels)
                # orig_images.append(images.view(batch_size, -1))

            # orig_images = torch.cat(orig_images, 0)
            # orig_images = orig_images.numpy()
            torch.cuda.empty_cache()
            model_poses = torch.cat(model_poses, 0)
            model_poses = model_poses.numpy()
            if model_orig_poses:
                model_orig_poses = torch.cat(model_orig_poses, 0)
                model_orig_poses = model_orig_poses.numpy()
            if model_presence:
                model_presence = torch.cat(model_presence, 0)
                model_presence = model_presence.numpy()
            targets = torch.cat(targets, 0)
            targets = targets.numpy().squeeze()

            for x, key in zip(
                    [orig_images, model_poses, model_orig_poses, model_presence],
                    ['images', 'poses', 'orig_poses', 'presence']
            ):
                if key == 'images':
                    continue

                if not len(x):
                    continue

                len_x = len(x)
                newx = []
                newtargets = []
                row1 = 0
                for rownum in range(len(x)):
                    row = x[rownum]
                    found = False
                    for newxrow in newx:
                        if all(newxrow == row):
                            found = True
                            break
                    if not found:
                        newx.append(row)
                        newtargets.append(targets[rownum])

                newx = np.array(newx)
                newtargets = np.array(newtargets)
                print('TSNEing the %s %s (%d)...' % (split, key, epoch))
                print('Orig len is %d / set length is %d / %d' % (len_x, len(newx), len(newtargets)))

                # embeddings = multiTSNE(
                #     n_jobs=4, n_components=2, perplexity=30, learning_rate=100.0
                # ).fit_transform(x)
                if use_cuda_tsne:
                    embeddings = cudaTSNE(
                        n_components=2, perplexity=30, learning_rate=100.0
                    ).fit_transform(newx)
                else:
                    embeddings = TSNE(
                        n_components=2, perplexity=30, learning_rate=100.0
                    ).fit_transform(newx)

                vis_x = embeddings[:, 0]
                vis_y = embeddings[:, 1]
                print(vis_x)
                print(vis_y)
                print(vis_x.shape, vis_y.shape, epoch)
                if any(np.isnan(vis_x)) or any(np.isnan(vis_y)):
                    print('\n***\n')
                    print('Nan!: ', key, np.isnan(x).any())
                    print(x)
                    print(newx)
                    if not np.isnan(x).any():
                        path_ = os.path.join(
                            path, 'tsnenan.%s.%s.%s.%03f.npy' % (
                                suffix, key, split, epoch)
                        )
                        with open(path_, 'wb') as f:
                            np.save(f, x)
                    print('\n***\n')
                else:
                    print('Carry on...')
                plt.scatter(vis_x, vis_y, c=newtargets, cmap=plt.cm.get_cmap("jet", 10), marker='.')
                plt.colorbar(ticks=range(10))
                plt.clim(-0.5, 9.5)
                path_ = os.path.join(path, 'tsne.%s.%s.%s%03d.png' % (
                    suffix, key, split, epoch))
                plt.savefig(path_)
                plt.clf()
                if comet_exp:
                    if split == 'train':
                        with comet_exp.train():
                            comet_exp.log_image(path_)
                    elif split == 'test':
                        with comet_exp.test():
                            comet_exp.log_image(path_)
                    # os.remove(path_)

            model_poses = []
            model_orig_poses = []
            model_presence = []
            orig_images = []
            targets = []


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_parameters(model):
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_loaders(args, rank=0):
    affnist_test_loader = None

    if args.dataset == 'MovingMNist':
        # NOTE: we are doing all single label here.
        train_set = MovingMNist(args.data_root, train=True, sequence=True, chance_single_label=1.)
        test_set = MovingMNist(args.data_root, train=False, sequence=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers)
        affnist_test_loader = None
    elif args.dataset == 'MovingMNist2':
        train_set = MovingMNist2(args.data_root, train=True, seq_len=5,
                                 image_size=64, colored=args.colored, tiny=False,
                                 is_triangle_loss='triangle' in args.criterion,
                                 is_reorder_loss='reorder' in args.criterion,
                                 num_digits=args.num_mnist_digits,
                                 center_start=args.fix_moving_mnist_center,
                                 single_angle=args.fix_moving_mnist_angle,
                                 step_length=args.step_length,
                                 positive_ratio=args.positive_ratio,
                                 use_simclr_xforms=args.use_simclr_xforms,
                                 use_diff_class_digit=args.use_diff_class_digit or args.criterion in ['probs_test'])
        test_set = MovingMNist2(args.data_root, train=False, seq_len=5,
                                image_size=64, colored=args.colored, tiny=False,
                                is_triangle_loss='triangle' in args.criterion,
                                is_reorder_loss='reorder' in args.criterion,
                                num_digits=args.num_mnist_digits,
                                center_start=args.fix_moving_mnist_center,
                                single_angle=args.fix_moving_mnist_angle,
                                step_length=args.step_length,
                                positive_ratio=args.positive_ratio,
                                use_diff_class_digit=args.use_diff_class_digit or args.criterion in ['probs_test'])
        # affnist_root = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/affnist'
        affnist_test_set = AffNist(
            args.affnist_root, train=False, subset=args.affnist_subset,
            transform=transforms.Compose([
                transforms.Pad(12),
                transforms.ToTensor(),
            ])
        )
        if args.num_gpus == 1:
            if args.use_class_sampler:
                assert(args.batch_size % 5 != 0)
                train_sampler = MovingMNISTClassSampler(
                    train_set, num_replicas=1, rank=0, shuffle=True)
                train_loader = torch.utils.data.DataLoader(
    	            dataset=train_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=True,
                    sampler=train_sampler,
                    drop_last=True
                )
            else:
                train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                           batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.num_workers)
        else:
            print('Distributed dataloader', rank, args.num_gpus, args.batch_size)
            if args.use_class_sampler:
                assert(args.batch_size % 5 != 0)
                train_sampler = MovingMNISTClassSampler(
                    train_set, num_replicas=args.num_gpus, rank=rank,
                    shuffle=True)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
    	            train_set,
    	            num_replicas=args.num_gpus,
    	            rank=rank
                )
            train_loader = torch.utils.data.DataLoader(
    	        dataset=train_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True
            )

        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers)
        affnist_test_loader = torch.utils.data.DataLoader(
            affnist_test_set, batch_size=args.batch_size, shuffle=False)            
    elif args.dataset == 'MovingMNist2.img1':
        train_set = MovingMNist2(args.data_root, train=True, seq_len=1, image_size=64,
                                 colored=args.colored, tiny=False,
                                 one_data_loop=True,
                                 is_triangle_loss='triangle' in args.criterion,
                                 num_digits=args.num_mnist_digits,
                                 center_start=args.fix_moving_mnist_center,
                                 single_angle=args.fix_moving_mnist_angle)
        test_set = MovingMNist2(args.data_root, train=False, seq_len=1, image_size=64,
                                colored=args.colored, tiny=False,
                                one_data_loop=True,
                                is_triangle_loss='triangle' in args.criterion,
                                num_digits=args.num_mnist_digits,
                                center_start=args.fix_moving_mnist_center,
                                single_angle=args.fix_moving_mnist_angle)
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers)
        affnist_root = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/affnist'
        affnist_test_set = AffNist(
            affnist_root, train=False, subset=False,
            transform=transforms.Compose([
                transforms.Pad(12),
                transforms.ToTensor(),
            ])
        )
        affnist_test_loader = torch.utils.data.DataLoader(
            affnist_test_set, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(
            args.data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize(40),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        test_set = torchvision.datasets.MNIST(
            args.data_root, train=False,
            transform=transforms.Compose([
                transforms.Resize(40),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == 'affnist':
        # NOTE: This was originally done with Pad(12) and RandomCrop(40) on
        # MNIST train, Pad(6) on MNIST test, and regular on AffNist.
        resize = args.resize
        mnist_padding = int((resize - 28)/2)
        affnist_padding = int((resize - 40)/2)

        train_set = torchvision.datasets.MNIST(
            args.data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(mnist_padding*2),
                transforms.RandomCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.1397,), (0.3081,))])
        )
        test_set = torchvision.datasets.MNIST(
            args.data_root, train=False,
            transform=transforms.Compose([
                transforms.Pad(mnist_padding),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        affnist_root = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/affnist'
        affnist_test_set = AffNist(
            affnist_root, train=False, subset=False,
            transform=transforms.Compose([
                transforms.Pad(affnist_padding),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False)
        affnist_test_loader = torch.utils.data.DataLoader(
            affnist_test_set, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'gymnastics':
        resize = args.resize
        transforms_augment_video = transforms.Compose([
            video_transforms.ToTensorVideo(),
            video_transforms.ResizeVideo((resize, resize), interpolation='nearest'),
            # video_transforms.RandomResizedCropVideo(224),
            # video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
        ])
        transforms_regular_video = transforms.Compose([
            video_transforms.ToTensorVideo(),
            # video_transforms.ResizeVideo((256, 256), interpolation='nearest'),
            # video_transforms.CenterCropVideo(224),
            video_transforms.ResizeVideo((resize, resize), interpolation='nearest'),
            # video_transforms.NormalizeVideo(mean=[0.485, 0.456, 0.406],
            #                                 std=[0.229, 0.224, 0.225]),
        ])

        train_set = GymnasticsVideo(
            transforms=transforms_regular_video, train=True,
            skip_videoframes=args.skip_videoframes,
            num_videoframes=args.num_videoframes,
            dist_videoframes=args.dist_videoframes,
            fps=args.fps, video_directory=args.gymnastics_video_directory,
            count_videos=args.count_videos, count_clips=args.count_clips
        )
        test_set = GymnasticsVideo(
            transforms=transforms_regular_video, train=True,
            skip_videoframes=args.skip_videoframes,
            num_videoframes=args.num_videoframes,
            dist_videoframes=args.dist_videoframes,
            fps=args.fps, video_directory=args.gymnastics_video_directory,
            count_videos=args.count_videos, count_clips=args.count_clips
        )
        print("Batch size: ", args.batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'gymnasticsRgbFrame':
        resize = args.resize
        transforms_augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])
        transforms_regular = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])
        train_set = GymnasticsRgbFrame(
            args.indices_file,
            transforms=transforms_augment,
            train=True,
            skip_videoframes=args.skip_videoframes,
            num_videoframes=args.num_videoframes,
            dist_videoframes=args.dist_videoframes,
            video_directory=args.gymnastics_video_directory,
            video_names=args.video_names.split(','),
            is_reorder_loss='reorder' in args.criterion,
            is_triangle_loss='triangle' in args.criterion,
            positive_ratio=args.positive_ratio,
            tau_min=args.min_distance,
            tau_max=args.max_distance
        )
        test_set = GymnasticsRgbFrame(
            args.indices_file,
            transforms=transforms_regular,
            train=False,
            skip_videoframes=args.skip_videoframes,
            num_videoframes=args.num_videoframes,
            dist_videoframes=args.dist_videoframes,
            video_directory=args.gymnastics_video_directory,
            video_names=args.video_names.split(','),
            is_reorder_loss='reorder' in args.criterion,
            is_triangle_loss='triangle' in args.criterion,
            positive_ratio=args.positive_ratio,
            tau_min=args.min_distance,
            tau_max=args.max_distance
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=gymnastics_flow_collate)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=gymnastics_flow_collate)
        print("Batch size: ", args.batch_size, ' Loader size: ',
              len(train_loader), len(test_loader))
    elif args.dataset == 'gymnastics_flow':
        resize = args.resize
        transforms_augment_video = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            # video_transforms.NormalizeVideo(
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transforms_regular_video = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            # video_transforms.NormalizeVideo(
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        gymnastics_root = '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion'
        train_set = gymnastics_flow(
            os.path.join(gymnastics_root, 'flows'),
            os.path.join(gymnastics_root, 'still-bad-flows.json'),
            os.path.join(gymnastics_root, 'nan_flow_list.json'),
            os.path.join(gymnastics_root, 'magnitude.npy'),
            os.path.join(gymnastics_root, 'file_dict.json'),
            range_size=5,
            transform=transforms_regular_video)
        test_set = gymnastics_flow(
            os.path.join(gymnastics_root, 'flows'),
            os.path.join(gymnastics_root, 'still-bad-flows.json'),
            os.path.join(gymnastics_root, 'nan_flow_list.json'),
            os.path.join(gymnastics_root, 'magnitude.npy'),
            os.path.join(gymnastics_root, 'file_dict.json'),
            range_size=5,
            transform=transforms_regular_video)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=gymnastics_flow_collate)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=gymnastics_flow_collate)
    elif args.dataset == 'GymnasticsFlowExperiment':
        transform_video = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.resize, args.resize)),
            transforms.ToTensor(),
            # video_transforms.NormalizeVideo(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        gymnastics_root = '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion'
        # Note: there is only 2628.fps25 in flow_imgs for now
        train_set = GymnasticsFlowExperiment(
            os.path.join(gymnastics_root, 'flows'),
            os.path.join(gymnastics_root, 'file_dict.json'),
            args.video_names,
            transform=transform_video,
            train=True,
            range_size=5,
            is_flow=False)
        test_set = GymnasticsFlowExperiment(
            os.path.join(gymnastics_root, 'flows'),
            os.path.join(gymnastics_root, 'file_dict.json'),
            args.video_names,
            transform=transform_video,
            train=False,
            range_size=5,
            is_flow=False)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=gymnastics_flow_collate)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=gymnastics_flow_collate)

    return train_loader, test_loader, affnist_test_loader

# Training
def train(epoch, step, net, optimizer, criterion, loader, args, device, comet_exp=None):
    net.train()
    if comet_exp:
        with comet_exp.train():
            comet_exp.log_current_epoch(epoch)

    averages = {
        'loss': Averager(),
        'grad_norm': Averager()
    }

    t = time.time()
    optimizer.zero_grad()

    if args.num_gpus > 1:
        loader.sampler.set_epoch(epoch)

    for batch_idx, (images, labels) in enumerate(loader):
        if criterion == 'triplet':
            # NOTE: Zeping.
            images = images.cuda(device)
            loss, stats = capsule_time_model.get_triplet_loss(net, images)
            averages['loss'].add(loss.item())
            positive_distance = stats['positive_distance']
            negative_distance = stats['negative_distance']
            extra_s = 'Pos distance: {:.5f} | Neg distance: {:.5f}'.format(
                positive_distance, negative_distance
            )
        elif criterion == 'bce':
            images = images.cuda(device)
            labels = labels.cuda(device)
            loss, stats = capsule_time_model.get_bce_loss(net, images, labels)
            averages['loss'].add(loss.item())
            true_positive_total += stats['true_pos']
            num_targets_total += stats['num_targets']
            extra_s = 'True Pos Rate: {:.5f} ({} / {}).'.format(
                100. * true_positive_total / num_targets_total,
                true_positive_total, num_targets_total
            )
        elif criterion == 'nce':
            images = images.cuda(device)
            loss, stats = capsule_time_model.get_nce_loss(net, images, args)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = 'Pos sim: {:.5f} | Neg sim: {:.5f}.'.format(
                averages['pos_sim'].item(), averages['neg_sim'].item()
            )
        elif criterion == 'xent':
            images = images.cuda(device)
            batch_size, num_images = images.shape[:2]
            # Change view so that we can put everything through the model at once.
            images = images.view(batch_size * num_images, *images.shape[2:])
            labels = labels.squeeze()
            labels = labels.cuda(device)
            loss, stats = capsule_time_model.get_xent_loss(net, images, labels)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = 'Acc: {:.5f}.'.format(
                averages['accuracy'].item()
            )
        elif criterion == 'reorder':
            if images is None and labels is None:
                continue
            images = images.cuda(device)
            labels = labels.cuda(device)
            loss, stats = capsule_time_model.get_reorder_loss(net, images, device, args, labels=labels)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                 for k, v in averages.items()])
            # del images
        elif criterion == 'reorder2':
            loss, stats = capsule_time_model.get_reorder_loss2(net, images, device, args, labels=labels)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                 for k, v in averages.items()])
        elif criterion in ['triangle', 'triangle_cos', 'triangle_margin',
                           'triangle_margin2', 'triangle_margin2_angle']:
            images = images.cuda(device)
            loss, stats = capsule_time_model.get_triangle_loss(net, images, device, args)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                 for k, v in averages.items()])
        elif criterion in ['triangle_margin2_angle_nce']:
            if not args.use_hinge_loss and not args.use_angle_loss:
                if random.random() < 0.5:
                    select_two = [0, 1]
                else:
                    select_two = [0, 2]
                images = images[:, select_two]
            images = images.cuda(device)
            loss, stats = capsule_time_model.get_triangle_nce_loss(net, images, device, epoch, args)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                 for k, v in averages.items() \
                                 if 'capsule_prob' not in k])
        elif criterion in ['discriminative_probs']:
            images = images[:, [0, 2]]
            images = images.cuda(device)
            labels = labels.squeeze()
            labels = labels.cuda(device)
            loss, stats = capsule_time_model.get_discriminative_probs(
                net, images, labels, device, epoch, args)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                 for k, v in averages.items() \
                                 if 'capsule_prob' not in k])
        elif criterion in ['probs_test']:
            images = images.cuda(device)
            loss, stats = capsule_time_model.get_probs_test_loss(
                net, images, device, epoch, args)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                 for k, v in averages.items() \
                                 if 'capsule_prob' not in k])
            
        loss.backward()

        total_norm = 0.
        for p in net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        averages['grad_norm'].add(total_norm ** (1. / 2))

        optimizer.step()
        optimizer.zero_grad()

        step += len(images) * args.num_gpus
        if comet_exp is not None:
            del images
            torch.cuda.empty_cache()

            if batch_idx % 25 == 0 and batch_idx > 0:
                log_text = ('Train Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
                print(log_text.format(epoch+1, batch_idx, len(loader),
                                      time.time() - t, averages['loss'].item())
                )
                t = time.time()
                with comet_exp.train():
                    epoch_avgs = {k: v.item() for k, v in averages.items()}
                    comet_exp.log_metrics(epoch_avgs, step=step, epoch=epoch)
                    
    train_loss = averages['loss'].item()
    train_acc = averages['accuracy'].item() if 'accuracy' in averages else None

    if comet_exp is not None:
        for key, value in stats.items():
            if key not in averages:
                averages[key] = Averager()
            averages[key].add(value)
        extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                             for k, v in averages.items()])
        log_text = ('Train Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
        print(log_text.format(epoch+1, batch_idx, len(loader),
                              time.time() - t, averages['loss'].item()))

        t = time.time()
        with comet_exp.train():
            epoch_avgs = {k: v.item() for k, v in averages.items()}
            comet_exp.log_metrics(epoch_avgs, step=step, epoch=epoch)

    return train_loss, train_acc, step


def test(epoch, step, net, criterion, loader, args, device, store_dir=None, comet_exp=None, is_affnist=False):
    net.eval()

    averages = {
        'loss': Averager()
    }

    t = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if criterion == 'triplet':
                # NOTE: Zeping.
                images = images.cuda(device)
                loss, stats = capsule_time_model.get_triplet_loss(net, images)
                averages['loss'].add(loss.item())
                positive_distance = stats['positive_distance']
                negative_distance = stats['negative_distance']
                extra_s = 'Pos distance: {:.5f} | Neg distance: {:.5f}'.format(
                    positive_distance, negative_distance
                )
            elif criterion == 'bce':
                images = images.cuda(device)
                labels = labels.cuda(device)
                loss, stats = capsule_time_model.get_bce_loss(net, images, labels)
                averages['loss'].add(loss.item())
                true_positive_total += stats['true_pos']
                num_targets_total += stats['num_targets']
                extra_s = 'True Pos Rate: {:.5f} ({} / {}).'.format(
                    100. * true_positive_total / num_targets_total,
                    true_positive_total, num_targets_total
                )
            elif criterion == 'nce':
                images = images.cuda(device)
                loss, stats = capsule_time_model.get_nce_loss(net, images, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = 'Pos sim: {:.5f} | Neg sim: {:.5f}.'.format(
                    averages['pos_sim'].item(), averages['neg_sim'].item()
                )
            elif criterion == 'xent':
                images = images.cuda(device)
                batch_size, num_images = images.shape[:2]
                # Change view so that we can put everything through the model at once.
                images = images.view(batch_size * num_images, *images.shape[2:])
                labels = labels.squeeze()
                labels = labels.cuda(device)
                loss, stats = capsule_time_model.get_xent_loss(net, images, labels)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = 'Acc: {:.5f}.'.format(
                    averages['accuracy'].item()
                )
            elif criterion == 'reorder':
                if images is None and labels is None:
                    continue
                images = images.cuda(device)
                labels = labels.cuda(device)
                loss, stats = capsule_time_model.get_reorder_loss(net, images, device, args, labels=labels)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items()])
            elif criterion == 'reorder2':
                loss, stats = capsule_time_model.get_reorder_loss2(net, images, device, args, labels=labels)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items()])
            elif criterion in ['triangle', 'triangle_cos', 'triangle_margin',
                               'triangle_margin2', 'triangle_margin2_angle']:
                images = images.cuda(device)
                loss, stats = capsule_time_model.get_triangle_loss(net, images, device, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items()])
            elif criterion in ['triangle_margin2_angle_nce']:
                if not args.use_hinge_loss and not args.use_angle_loss:
                    if random.random() < 0.5:
                        select_two = [0, 1]
                    else:
                        select_two = [0, 2]
                    images = images[:, select_two]
                images = images.cuda(device)
                loss, stats = capsule_time_model.get_triangle_nce_loss(net, images, device, epoch, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items() \
                                     if 'capsule_prob' not in k])
            elif criterion in ['discriminative_probs']:
                if not is_affnist:
                    images = images[:, [0, 2]]
                images = images.cuda(device)
                labels = labels.squeeze()
                labels = labels.cuda(device)
                loss, stats = capsule_time_model.get_discriminative_probs(
                    net, images, labels, device, epoch, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items() \
                                     if 'capsule_prob' not in k])
            elif criterion in ['probs_test']:
                images = images.cuda(device)
                loss, stats = capsule_time_model.get_probs_test_loss(
                    net, images, device, epoch, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items() \
                                     if 'capsule_prob' not in k])

            if comet_exp and batch_idx % 100 == 0 and batch_idx > 0:
                log_text = ('Val Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
                print(log_text.format(epoch + 1, batch_idx, len(loader),
                                      time.time() - t, averages['loss'].item())
                )
                t = time.time()

        if comet_exp:
            del images
            torch.cuda.empty_cache()

            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                 for k, v in averages.items()])
            log_text = ('Val Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
            print(log_text.format(epoch + 1, batch_idx, len(loader),
                                  time.time() - t, averages['loss'].item())
            )

            test_loss = averages['loss'].item()
            with comet_exp.test():
                epoch_avgs = {k: v.item() for k, v in averages.items()}
                comet_exp.log_metrics(epoch_avgs, step=step, epoch=epoch)

    test_acc = averages['accuracy'].item() if 'accuracy' in averages else None
    return test_loss, test_acc


def main(gpu, args, port=12355):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend='nccl', rank=gpu, world_size=args.num_gpus)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = getattr(configs, args.config).config
    print(config)

    train_loader, test_loader, affnist_test_loader = get_loaders(args, rank=gpu)

    print('==> Building model..')
    num_frames = 4 if args.criterion == 'reorder2' else 3
    is_discriminating_model = all([
        'reorder' not in args.criterion, 'triangle' not in args.criterion,
        args.criterion != 'discriminative_probs'
    ])
        
    if args.use_resnet:
        if args.criterion not in ['reorder', 'reorder2']:
            raise
        net = ReorderResNet(resnet_type=args.resnet_type, pretrained=args.resnet_pretrained)
    elif args.criterion == 'probs_test':
        net = capsule_time_model.ProbsTest(
            64, config['params']['class_capsules']['num_caps'],
            temperature=args.presence_temperature,
            use_noise=args.use_rand_presence_noise)
    else:
        do_capsule_computation = args.criterion not in [
            'triangle_margin2_angle_nce', 'discriminative_probs'] or \
            args.use_nce_loss or args.use_hinge_loss or args.use_angle_loss
        net = capsule_time_model.CapsTimeModel(
            config['params'],
            args.backbone,
            args.dp,
            args.num_routing,
            sequential_routing=args.sequential_routing,
            num_frames=num_frames,
            is_discriminating_model=is_discriminating_model,
            use_presence_probs=args.use_presence_probs,
            presence_temperature=args.presence_temperature,
            presence_loss_type=args.presence_loss_type,
            do_capsule_computation=do_capsule_computation,
            do_discriminative_probs=args.criterion == 'discriminative_probs'
        )

    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(),
                              lr=args.lr,
                              momentum=0.9,
                              weight_decay=args.weight_decay)

    if args.use_scheduler:
        # CIFAR used milestones of [150, 250] with gamma of 0.1
        milestones = [int(k) for k in args.schedule_milestones.split(',')]
        gamma = args.schedule_gamma
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=milestones,
                                                         gamma=gamma)
    else:
        scheduler = None

    total_params = count_parameters(net)
    print('Total Params %d' % total_params)

    today = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    if args.name and args.counter is not None:
        store_dir = os.path.join(args.results_dir, args.name,
                                 str(args.counter), today)
    else:
        store_dir = os.path.join(args.results_dir, today)
    if gpu == 0 and not os.path.isdir(store_dir) and not args.debug:
        os.makedirs(store_dir)

    torch.cuda.set_device(gpu)
    net.cuda(gpu)
    net = DDP(net, device_ids=[gpu], find_unused_parameters=True)

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if args.resume_dir and not args.debug:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.pth'))
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    results = {
        'total_params': total_params,
        'args': args,
        'params': config['params'],
        'affnist_acc': [],
        'train_acc': [],
        'test_acc': [],
        'affnist_loss': [],
        'train_loss': [],
        'test_loss': [],
    }

    total_epochs = args.epoch
    if not args.debug:
        store_file = 'dataset_%s_num_routing_%s_backbone_%s.dct' % (
            str(args.dataset), str(args.num_routing), args.backbone
        )
        store_file = os.path.join(store_dir, store_file)

    print(args.dataset)

    if args.use_comet and gpu == 0:
        comet_exp = CometExperiment(api_key="hIXq6lDzWzz24zgKv7RYz6blo",
                                    project_name="capsules",
                                    workspace="cinjon",
                                    auto_metric_logging=True,
                                    auto_output_logging=None,
                                    auto_param_logging=False)
        comet_params = vars(args)
        user = 'cinjon' if getpass.getuser() == 'resnick' else 'zeping'
        comet_params['user_name'] = user
        if args.counter is not None and args.name is not None:
            comet_exp.set_name('%s-%d' % (args.name, args.counter))
        comet_exp.log_parameters(vars(args))
    else:
        comet_exp = None

    torch.autograd.set_detect_anomaly(True)

    # test_loss, test_acc = test(0, net, args.criterion, test_loader, args, best_negative_distance, device, store_dir=store_dir)
    # print('Before staarting Test Acc: ', test_acc)
    # affnist_loss, affnist_acc = test(0, net, args.criterion, affnist_test_loader, args, best_negative_distance, device, store_dir=store_dir)
    # print('Before starting affnist_acc: ', affnist_acc)


    step = 0
    last_test_loss = 1e8
    last_saved_epoch = 0
    device = gpu

    # if gpu == 0 and 'triangle' not in args.criterion and \
    #    args.dataset in ['affnist', 'MovingMNist2', 'MovingMNist2.img1']:
    #     affnist_loss, affnist_acc = test(
    #         0, step, net, args.criterion, affnist_test_loader, args,
    #         device, store_dir=store_dir, comet_exp=comet_exp, is_affnist=True)

    # if gpu == 0:
    #     run_tsne(
    #         args.data_root, net, store_dir, 0, use_moving_mnist=True,
    #         comet_exp=comet_exp, center_start=args.fix_moving_mnist_center,
    #         single_angle=args.fix_moving_mnist_angle,
    #         use_cuda_tsne=args.use_cuda_tsne, tsne_batch_size=args.tsne_batch_size)

    for epoch in range(start_epoch, start_epoch + total_epochs):
        print('Starting Epoch %d' % epoch)
        train_loss, train_acc, step = train(epoch, step, net, optimizer, args.criterion, train_loader, args, device, comet_exp)
        # if train_acc is not None:
        #     print('Train Acc %.4f.' % train_acc)
        #     results['train_acc'].append(train_acc)
        # else:
        #     print('Train Loss %.4f.' % train_loss)
        #     results['train_loss'].append(train_loss)

        if scheduler:
            scheduler.step()

        if epoch % 3 == 0 and gpu == 0:
            test_loss, test_acc = test(
                epoch, step, net, args.criterion, test_loader, args, device,
                store_dir=store_dir, comet_exp=comet_exp)
            # if test_acc is not None:
            #     print('Test Acc %.4f.' % test_acc)
            #     results['test_acc'].append(test_acc)
            # else:
            #     print('Test Loss %.4f.' % test_loss)
            #     results['test_loss'].append(test_loss)

            if not args.debug and epoch >= last_saved_epoch + 10 and \
               last_test_loss > test_loss:
                state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'loss': test_loss
                }
                print('\n***\nSaving epoch %d\n***\n' % epoch)
                torch.save(state, os.path.join(store_dir, 'ckpt.epoch%d.pth' % epoch))
                last_test_loss = test_loss
                last_saved_epoch = epoch

            print('test_acc: ', test_acc, test_loss)
            print('trainacc: ', train_acc, train_loss)
            if gpu == 0 and 'triangle' not in args.criterion and \
               'probs_test' not in args.criterion and \
               args.dataset in ['affnist', 'MovingMNist2', 'MovingMNist2.img1'] and \
               test_acc >= .95 and train_acc >= .95 and epoch > 0:
                print('\n***\nRunning affnist...')
                affnist_loss, affnist_acc = test(
                    epoch, step, net, args.criterion, affnist_test_loader, args,
                    device, store_dir=store_dir, comet_exp=comet_exp, is_affnist=True)
                results['affnist_acc'].append(affnist_acc)
                print('AFFNIST Results: ', affnist_loss, affnist_acc)

        if comet_exp:
            comet_exp.log_epoch_end(epoch)

        if gpu == 0 and args.do_mnist_test_every and epoch % args.do_mnist_test_every == 0 and epoch > args.do_mnist_test_after:
            print('\n***\nStarting MNist Test (%d)\n***' % epoch)
            linpred_train.run_ssl_model(
                epoch, net, args.data_root, args.affnist_root, comet_exp,
                args.mnist_batch_size, args.colored, args.num_workers,
                config['params'], args.backbone, args.num_routing, num_frames,
                args.mnist_lr, args.mnist_weight_decay, args.affnist_subset)
            print('\n***\nEnded MNist Test (%d)\n***' % epoch)

        if all([
                gpu == 0,
                args.do_tsne_test_every is not None,
                epoch % args.do_tsne_test_every == 0,
                epoch > args.do_tsne_test_after
        ]):
            print('\n***\nStarting TSNE (%d)\n***' % epoch)
            run_tsne(
                args.data_root, net, store_dir, epoch, use_moving_mnist=True,
                comet_exp=comet_exp, center_start=args.fix_moving_mnist_center,
                single_angle=args.fix_moving_mnist_angle,
                use_cuda_tsne=args.use_cuda_tsne, tsne_batch_size=args.tsne_batch_size)
            print('\n***\nEnded TSNE (%d)\n***' % epoch)

        if gpu == 0 and not args.debug:
            with open(store_file, 'wb') as f:
                pickle.dump(results, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training Capsules using Inverted Dot-Product Attention Routing'
    )

    parser.add_argument('--mode',
                        default='run',
                        type=str,
                        help='mode to use, either run or jobarray')
    parser.add_argument('--counter', type=int, default=None,
                        help='the counter when running with slurm')
    parser.add_argument('--name', type=str, default=None,
                        help='the name when running with slurm')
    parser.add_argument('--resume_dir',
                        '-r',
                        default='',
                        type=str,
                        help='dir where we resume from checkpoint')
    parser.add_argument('--no_use_comet',
                        action='store_true',
                        help='use comet or not')
    parser.add_argument('--do_mnist_test_every',
                        default=None, type=int,
                        help='if an integer > 0, then do the mnist_affnist test ' \
                        'every this many epochs.')
    parser.add_argument('--do_mnist_test_after',
                        default=10, type=int,
                        help='if an integer > 0, then do the mnist_affnist test ' \
                        'starting at this epoch.')
    parser.add_argument('--do_tsne_test_every',
                        default=None, type=int,
                        help='if an integer > 0, then do the mnist_affnist test ' \
                        'every this many epochs.')
    parser.add_argument('--do_tsne_test_after',
                        default=10, type=int,
                        help='if an integer > 0, then do the mnist_affnist test ' \
                        'starting at this epoch.')
    parser.add_argument('--mnist_batch_size',
                        default=16,
                        type=int,
                        help='number of batch size')
    parser.add_argument('--mnist_lr',
                        default=0.1,
                        type=float,
                        help='number of batch size')
    parser.add_argument('--mnist_weight_decay',
                        default=5e-4,
                        type=float,
                        help='weight decay')
    parser.add_argument('--num_routing',
                        default=1,
                        type=int,
                        help='number of routing. Recommended: 0,1,2,3.')
    parser.add_argument('--dataset',
                        default='MovingMNist2',
                        type=str,
                        help='dataset: MovingMNist and a host of gymnastics ones.')
    parser.add_argument('--use_class_sampler',
                        action='store_true',
                        help='whether to use the class sampler with MovingMNist2')
    parser.add_argument('--use_diff_class_digit',
                        action='store_true',
                        help='whether to use the class sampler with MovingMNist2')
    parser.add_argument('--backbone',
                        default='resnet',
                        type=str,
                        help='type of backbone. simple or resnet')
    parser.add_argument('--data_root',
                        default='/misc/kcgscratch1/ChoGroup/resnick/vidcaps/MovingMNist/',
                        type=str,
                        help='root of where to put the data.')
    parser.add_argument('--results_dir', type=str, default='./contrastive_results',
                        help='whether to store the results')
    parser.add_argument('--num_workers',
                        default=2,
                        type=int,
                        help='number of workers. 0 or 2')
    parser.add_argument('--num_gpus', default=1, type=int, help='number of gpus.')
    parser.add_argument('--config',
                        default='resnet_backbone_movingmnist',
                        type=str,
                        help='path of the config')
    parser.add_argument('--debug',
                        action='store_true',
                        help='use debug mode (without saving to a directory)')
    parser.add_argument('--sequential_routing',
                        action='store_true',
                        help='not using concurrent_routing')
    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help='learning rate. 0.1 for SGD, 1e-2 for Adam (trying)')
    parser.add_argument('--dp', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--weight_decay',
                        default=0, # was 5e-4 but default off
                        type=float,
                        help='weight decay')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--epoch', default=350, type=int, help='number of epoch')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='number of batch size')
    parser.add_argument('--optimizer',
                        default='adam',
                        type=str,
                        help='adam or sgd.')
    parser.add_argument('--criterion',
                        default='triplet',
                        type=str,
                        help='triplet, nce, bce, xent, or reorder.')
    parser.add_argument('--nce_positive_frame_num',
                        default=10,
                        type=int,
                        help='the # of frames from anchor to use as positive.')
    parser.add_argument('--nce_temperature',
                        default=1.0,
                        type=float,
                        help='temperature to multiply with the similarity')
    parser.add_argument('--nce_presence_temperature',
                        default=1.0,
                        type=float,
                        help='temperature to multiply with the similarity')
    parser.add_argument('--nce_lambda',
                        default=1.0,
                        type=float,
                        help='lambda on the nce loss.')
    parser.add_argument('--nce_presence_lambda',
                        default=1.0,
                        type=float,
                        help='lambda on the nce loss.')
    parser.add_argument('--use_random_anchor_frame',
                        default=False,
                        action='store_true',
                        help='whether to use a random anchor frame')
    parser.add_argument('--use_scheduler',
                        action='store_true',
                        help='whether to use the scheduler or not.')
    parser.add_argument('--schedule_milestones', type=str, default='150,250',
                        help='the milestones in the LR.')
    parser.add_argument('--schedule_gamma', type=float, default=0.1,
                        help='the default LR gamma.')
    parser.add_argument('--affnist_root',
                        type=str,
                        default='/misc/kcgscratch1/ChoGroup/resnick/vidcaps/affnist',
                        help='affnist data root')
    parser.add_argument('--affnist_subset',
                        default=False,
                        action='store_true',
                        help='whether to use subset of affnist or not')

    # Reordering
    parser.add_argument('--lambda_ordering',
                        default=1.,
                        type=float,
                        help='the lambda on the ordering loss')
    parser.add_argument('--lambda_object_sim',
                        default=1.,
                        type=float,
                        help='the lambda on the objects being similar.')
    parser.add_argument('--lambda_sparse_presence',
                        default=1.,
                        type=float,
                        help='the lambda on the presence L1.')
    parser.add_argument('--lambda_between_entropy',
                        default=1.,
                        type=float,
                        help='the lambda on the between entropy.')
    parser.add_argument('--lambda_within_entropy',
                        default=1.,
                        type=float,
                        help='the lambda on the within entropy.')

    # Triangle
    parser.add_argument('--triangle_lambda',
                        default=1.,
                        type=float,
                        help='the lambda on the distance between first and third frames.')
    parser.add_argument('--triangle_cos_lambda',
                        default=1.,
                        type=float,
                        help='the lambda on the cos between first and third frames.')
    parser.add_argument('--triangle_margin_lambda',
                        default=1.,
                        type=float,
                        help='the lambda on the margin between first and third frames.')
    parser.add_argument('--margin_gamma',
                        default=1.,
                        type=float,
                        help='the gamma margin on how minimal we want ac')
    parser.add_argument('--margin_gamma2',
                        default=.5,
                        type=float,
                        help='the gamma margin on how minimal we want ab/bc.')

    # Presence
    parser.add_argument('--use_presence_probs',
                        default=False,
                        action='store_true',
                        help='whether to use the presence probs from the pre_caps')
    parser.add_argument('--presence_temperature',
                        default=1.,
                        type=float,
                        help='the lambda on the presence L1.')
    parser.add_argument('--use_rand_presence_noise',
                        action='store_true')
    parser.add_argument(
        '--presence_loss_type',
        default='sigmoid_l1',
        type=str,
        help='the type of presence loss: sigmoid_l1, sigmoid_prior_sparsity')
    parser.add_argument('--no_use_angle_loss',
                        default=False,
                        action='store_true',
                        help='whether to use the angle loss in triangle_nce')
    parser.add_argument('--no_use_hinge_loss',
                        default=False,
                        action='store_true',
                        help='whether to use the hinge loss in triangle_nce')
    parser.add_argument('--no_use_nce_loss',
                        default=False,
                        action='store_true',
                        help='whether to use the nce loss in triangle_nce')
    parser.add_argument('--use_simclr_xforms',
                        action='store_true',
                        help='whether to use the transforms')
    parser.add_argument('--use_nce_probs',
                        default=False,
                        action='store_true',
                        help='whether to use the nce over the presence probs')
    parser.add_argument('--use_simclr_nce',
                        default=False,
                        action='store_true',
                        help='whether to use the simclr nce or ours.')
    parser.add_argument('--num_output_classes',
                        default=10,
                        type=float,
                        help='the number of classes. this is a hack and dumb.')
    parser.add_argument('--presence_samecos_lambda',
                        default=3.,
                        type=float,
                        help='the lambda on the cos between first and third frames.')
    parser.add_argument('--presence_diffcos_lambda',
                        default=3.,
                        type=float,
                        help='the lambda on the cos between first and third frames.')
    parser.add_argument('--presence_hinge_lambda',
                        default=1.,
                        type=float,
                        help='the lambda on the cos between first and third frames.')

    # VideoClip info
    parser.add_argument('--num_videoframes', type=int, default=100)
    parser.add_argument('--dist_videoframes', type=int, default=50, help='the frame interval between each sequence.')
    parser.add_argument('--resize', type=int, default=128, help='to what to resize the frames of the video.')
    parser.add_argument('--fps', type=int, default=5, help='the fps for the loaded VideoClips')
    parser.add_argument('--count_videos', type=int, default=32, help='the number of video fiels to includuek')
    parser.add_argument('--count_clips', type=int, default=-1, help='the number of clips to includue')
    parser.add_argument('--positive_ratio', type=float, default=0.5, help='positive ratio of reorder sequence')
    parser.add_argument('--min_distance', type=float, default=15, help='min(|a - b|, |d - e|)')
    parser.add_argument('--max_distance', type=float, default=60, help='|b - d|')
    parser.add_argument(
        '--skip_videoframes',
        type=int,
        default=5,
        help='the number of video frames to skip in between each one. using 1 means that there is no skip.'
    )
    parser.add_argument('--video_names',
                        default='2628',
                        type=str,
                        help='a comma separated list of videos to use for gymnastics. e.g. 1795,1016')
    parser.add_argument('--gymnastics_video_directory',
                        default='/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/feb102020',
                        type=str,
                        help='video directory of gymnastics files.')
    parser.add_argument('--use_resnet',
                        default=False,
                        action='store_true',
                        help='whether to use resnet to do reorder task',)
    parser.add_argument('--resnet_type',
                        default='resnet18',
                        type=str,
                        help='resnet18 or resnet50')
    parser.add_argument('--resnet_pretrained',
                        default=False,
                        action='store_true',
                        help='whether to use pretrained resnet or not')
    parser.add_argument('--indices_file',
                        default='/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/abc/skipframe_1_indices.json',
                        type=str,
                        help='path of the GymnasticsRgbFrame indices file',)

    # MovingMNist 2 info
    parser.add_argument('--no_colored', action='store_true',
                        help='whether to not use colored mnist')
    parser.add_argument('--fix_moving_mnist_center', action='store_true',
                        help='whether to fix the center of moving mnist.')
    parser.add_argument('--fix_moving_mnist_angle', action='store_true',
                        help='whether to fix the angle of moving mnist.')
    parser.add_argument('--num_mnist_digits', type=int, default=2,
                        help='the number of digits to use in moving mnist.')
    parser.add_argument('--step_length', type=float, default=0.035,
                        help='step length in movingmnist2 sequence')

    # TSNE info
    parser.add_argument('--use_cuda_tsne', action='store_true',
                        help='whether to use cudaTSNE or TSNE from sklearn')
    parser.add_argument('--tsne_batch_size', type=int, default=72,
                        help='batch size for tsne dataloader')

    args = parser.parse_args()
    if args.mode == 'jobarray':
        jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        user = 'cinjon' if getpass.getuser() == 'resnick' else 'zeping'
        if user == 'cinjon':
            counter, job = cinjon_jobs.run(find_counter=jobid)
        elif user == 'zeping':
            counter, job = zeping_jobs.run(find_counter=jobid)
        else:
            raise

        for key, value in job.items():
            setattr(args, key, value)
        print(counter, job, '\n', args, '\n')

    args.use_comet = (not args.no_use_comet) and (not args.debug)
    assert args.num_routing > 0
    args.colored = not args.no_colored
    args.use_angle_loss = not args.no_use_angle_loss
    args.use_hinge_loss = not args.no_use_hinge_loss
    args.use_nce_loss = not args.no_use_nce_loss

    # main(args)

    default_port = random.randint(10000, 19000)
    mp.spawn(main, nprocs=args.num_gpus, args=(args, default_port)) 
