"""

Sample Command:
python train.py --criterion bce --results_dir /.../resnick/vidcaps/results \
--debug --data_root /.../resnick/vidcaps/MovingMNist --batch_size 32 \
--config resnet_backbone_movingmnist2
"""

import os
import json
import time
import pickle
import random
import argparse
import numpy as np
from datetime import datetime

from comet_ml import Experiment as CometExperiment, OfflineExperiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import configs
from src.moving_mnist.moving_mnist import MovingMNist
from src import capsule_model
from src import capsule_time_model
from src.affnist import AffNist
from src.gymnastics import Gymnastics
from src.gymnastics.gymnastics_flow import gymnastics_flow, gymnastics_flow_collate
import video_transforms


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


def get_loaders(args):
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
        train_set = torchvision.datasets.MNIST(
            args.data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(12),
                transforms.RandomCrop(40),
                transforms.ToTensor(),
                transforms.Normalize((0.1397,), (0.3081,))])
        )
        test_set = torchvision.datasets.MNIST(
            args.data_root, train=False,
            transform=transforms.Compose([
                transforms.Pad(6),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        affnist_root = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/affnist'
        affnist_test_set = AffNist(
            affnist_root, train=False, subset=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=192, shuffle=False)
        affnist_test_loader = torch.utils.data.DataLoader(
            affnist_test_set, batch_size=192, shuffle=False)
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

        train_set = Gymnastics(
            transforms=transforms_regular_video, train=True,
            skip_videoframes=args.skip_videoframes,
            num_videoframes=args.num_videoframes,
            dist_videoframes=args.dist_videoframes,
            fps=args.fps, video_directory=args.gymnastics_video_directory,
            count_videos=args.count_videos, count_clips=args.count_clips
        )
        test_set = Gymnastics(
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
            train_set, batch_size=args.batch_size, shuffle=True, collate_fn=gymnastics_flow_collate)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, collate_fn=gymnastics_flow_collate)

    return train_loader, test_loader, affnist_test_loader


# Training
def train(epoch, step, net, optimizer, criterion, loader, args, device, comet_exp=None):
    net.train()
    if comet_exp:
        with comet_exp.train():
            comet_exp.log_current_epoch(epoch)

    total_positive_distance = 0.
    total_negative_distance = 0.

    averages = {
        'loss': Averager()
    }
    if criterion == 'nce':
        averages['pos_sim'] = Averager()
        averages['neg_sim'] = Averager()
    elif criterion == 'bce':
        averages['true_pos'] = Averager()
        averages['num_targets'] = Averager()
        true_positive_total = 0
        num_targets_total = 0
    elif criterion in ['xent', 'reorder']:
        averages['accuracy'] = Averager()
        averages['objects_sim_loss'] = Averager()
        averages['presence_sparsity_loss'] = Averager()
        averages['ordering_loss'] = Averager()

    t = time.time()
    optimizer.zero_grad()
    for batch_idx, (images, labels) in enumerate(loader):
        # images = images.to(device)

        if criterion == 'triplet':
            # NOTE: Zeping.
            images = images.to(device)
            loss, stats = net.get_triplet_loss(images)
            averages['loss'].add(loss.item())
            positive_distance = stats['positive_distance']
            negative_distance = stats['negative_distance']
            total_positive_distance += positive_distance
            total_negative_distance += negative_distance
            extra_s = 'Pos distance: {:.5f} | Neg distance: {:.5f}'.format(
                positive_distance, negative_distance
            )
        elif criterion == 'bce':
            images = images.to(device)
            labels = labels.to(device)
            loss, stats = net.get_bce_loss(images, labels)
            averages['loss'].add(loss.item())
            true_positive_total += stats['true_pos']
            num_targets_total += stats['num_targets']
            extra_s = 'True Pos Rate: {:.5f} ({} / {}).'.format(
                100. * true_positive_total / num_targets_total,
                true_positive_total, num_targets_total
            )
        elif criterion == 'nce':
            images = images.to(device)
            loss, stats = net.get_nce_loss(images, args)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                averages[key].add(value)
            extra_s = 'Pos sim: {:.5f} | Neg sim: {:.5f}.'.format(
                averages['pos_sim'].item(), averages['neg_sim'].item()
            )
        elif criterion == 'xent':
            images = images.to(device)
            labels = labels.to(device)
            loss, stats = net.get_xent_loss(images, labels)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                averages[key].add(value)
            extra_s = 'Acc: {:.5f}.'.format(
                averages['accuracy'].item()
            )
        elif criterion == 'reorder':
            loss, stats = net.get_reorder_loss(images, device, args)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                averages[key].add(value)
            extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                 for k, v in averages.items()])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += len(images)
        if batch_idx % 10 == 0 and batch_idx > 0:
            log_text = ('Train Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
            print(log_text.format(epoch+1, batch_idx, len(loader),
                                  time.time() - t, averages['loss'].item())
            )
            t = time.time()
            if comet_exp:
                with comet_exp.train():
                    epoch_avgs = {k: v.item() for k, v in averages.items()}
                    comet_exp.log_metrics(epoch_avgs, step=step, epoch=epoch)

    train_loss = averages['loss'].item()
    train_acc = averages['accuracy'].item()

    if comet_exp:
        with comet_exp.train():
            epoch_avgs = {k: v.item() for k, v in averages.items()}
            comet_exp.log_metrics(epoch_avgs, step=step, epoch=epoch)

    return train_loss, train_acc, step


def test(epoch, step, net, criterion, loader, args, best_negative_distance, device, store_dir=None, comet_exp=None):
    net.eval()
    total_positive_distance = 0.
    total_negative_distance = 0.

    averages = {
        'loss': Averager()
    }
    if criterion == 'nce':
        averages['pos_sim'] = Averager()
        averages['neg_sim'] = Averager()
    elif criterion == 'bce':
        averages['true_pos'] = Averager()
        averages['num_targets'] = Averager()
        true_positive_total = 0
        num_targets_total = 0
    elif criterion in ['xent', 'reorder']:
        averages['accuracy'] = Averager()
        averages['objects_sim_loss'] = Averager()
        averages['presence_sparsity_loss'] = Averager()
        averages['ordering_loss'] = Averager()

    t = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            # images = images.to(device)

            if criterion == 'triplet':
                # NOTE: Zeping.
                images = images.to(device)
                loss, stats = net.get_triplet_loss(images)
                averages['loss'].add(loss.item())
                positive_distance = stats['positive_distance']
                negative_distance = stats['negative_distance']
                total_positive_distance += positive_distance
                total_negative_distance += negative_distance
                extra_s = 'Pos distance: {:.5f} | Neg distance: {:.5f}'.format(
                    positive_distance, negative_distance
                )
            elif criterion == 'bce':
                images = images.to(device)
                labels = labels.to(device)
                loss, stats = net.get_bce_loss(images, labels)
                averages['loss'].add(loss.item())
                true_positive_total += stats['true_pos']
                num_targets_total += stats['num_targets']
                extra_s = 'True Pos Rate: {:.5f} ({} / {}).'.format(
                    100. * true_positive_total / num_targets_total,
                    true_positive_total, num_targets_total
                )
            elif criterion == 'nce':
                images = images.to(device)
                loss, stats = net.get_nce_loss(images, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    averages[key].add(value)
                extra_s = 'Pos sim: {:.5f} | Neg sim: {:.5f}.'.format(
                    averages['pos_sim'].item(), averages['neg_sim'].item()
                )
            elif criterion == 'xent':
                images = images.to(device)
                loss, stats = net.get_xent_loss(images, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    averages[key].add(value)
                extra_s = 'Acc: {:.5f}.'.format(
                    averages['accuracy'].item()
                )
            elif criterion == 'reorder':
                loss, stats = net.get_reorder_loss(images, device, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items()])

            if batch_idx % 100 == 0:
                log_text = ('Val Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
                print(log_text.format(epoch + 1, batch_idx, len(loader),
                                      time.time() - t, averages['loss'].item())
                )
                t = time.time()

        test_loss = averages['loss'].item()
        if comet_exp:
            with comet_exp.test():
                epoch_avgs = {k: v.item() for k, v in averages.items()}
                comet_exp.log_metrics(epoch_avgs, step=step, epoch=epoch)

    # Save checkpoint.
    state = {
        'net': net.state_dict(),
        'total_positive_distance': total_positive_distance,
        'total_negative_distance': total_negative_distance,
        'epoch': epoch,
        'loss': test_loss
    }

    if not store_dir:
        print('Do not have store_dir!')
    elif not args.debug:
        print('\n***\nSaving epoch %d\n***\n' % epoch)
        torch.save(state, os.path.join(store_dir, 'ckpt.epoch%d.pth' % epoch))

    test_acc = averages['accuracy'].item()
    return test_loss, test_acc


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    config = getattr(configs, args.config).config
    print(config)

    train_loader, test_loader, affnist_test_loader = get_loaders(args)

    print('==> Building model..')
    if args.criterion == 'reorder':
        net = capsule_time_model.CapsTimeModel(config['params'],
                                               args.backbone,
                                               args.dp,
                                               args.num_routing,
                                               sequential_routing=args.sequential_routing)
    else:
        net = capsule_model.CapsModel(config['params'],
                                      args.backbone,
                                      args.dp,
                                      args.num_routing,
                                      sequential_routing=args.sequential_routing)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=args.lr,
                               # weight_decay=args.weight_decay
        )
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
    store_dir = os.path.join(args.results_dir, today)
    if not os.path.isdir(store_dir) and not args.debug:
        os.makedirs(store_dir)

    net = net.to(device)
    if device == 'cuda':
        if args.num_gpus > 1:
            net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    best_negative_distance = 0
    if args.resume_dir and not args.debug:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.pth'))
        net.load_state_dict(checkpoint['net'])
        best_negative_distance = checkpoint.get('best_negative_distance', 0)
        start_epoch = checkpoint['epoch']

    results = {
        'total_params': total_params,
        'args': args,
        'params': config['params'],
        'affnist_acc': [],
        'train_acc': [],
        'test_acc': [],
    }

    total_epochs = args.epoch
    if not args.debug:
        store_file = 'dataset_%s_num_routing_%s_backbone_%s.dct' % (
            str(args.dataset), str(args.num_routing), args.backbone
        )
        store_file = os.path.join(store_dir, store_file)

    print(args.dataset)

    if args.use_comet:
        comet_exp = CometExperiment(api_key="hIXq6lDzWzz24zgKv7RYz6blo",
                                    project_name="capsules",
                                    workspace="cinjon",
                                    auto_metric_logging=True,
                                    auto_output_logging=None,
                                    auto_param_logging=False)
        comet_exp.log_parameters(vars(args))
    else:
        comet_exp = None


    # test_loss, test_acc = test(0, net, args.criterion, test_loader, args, best_negative_distance, device, store_dir=store_dir)
    # print('Before staarting Test Acc: ', test_acc)
    # affnist_loss, affnist_acc = test(0, net, args.criterion, affnist_test_loader, args, best_negative_distance, device, store_dir=store_dir)
    # print('Before starting affnist_acc: ', affnist_acc)

    step = 0
    for epoch in range(start_epoch, start_epoch + total_epochs):
        print('Starting Epoch %d' % epoch)
        train_loss, train_acc, step = train(epoch, step, net, optimizer, args.criterion, train_loader, args, device, comet_exp)
        print('Train Acc %.4f.' % train_acc)
        results['train_acc'].append(train_acc)

        if scheduler:
            scheduler.step()

        if args.dataset != 'gymnastics':
            test_loss, test_acc = test(epoch, step, net, args.criterion, test_loader, args, best_negative_distance, device, store_dir=store_dir, comet_exp=comet_exp)
            print('Test Acc %.4f.' % test_acc)
            results['test_acc'].append(test_acc)

        if args.dataset == 'affnist' and test_acc >= .9875 and train_acc >= .9875 and epoch > 0:
            print('\n***\nRunning affnist...')
            affnist_loss, affnist_acc = test(epoch, net, args.criterion, affnist_test_loader, args, best_negative_distance, device, store_dir=store_dir)
            results['affnist_acc'].append(affnist_acc)

        if comet_exp:
            comet_exp.log_epoch_end(epoch)

        if not args.debug:
            with open(store_file, 'wb') as f:
                pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training Capsules using Inverted Dot-Product Attention Routing'
    )

    parser.add_argument('--resume_dir',
                        '-r',
                        default='',
                        type=str,
                        help='dir where we resume from checkpoint')
    parser.add_argument('--no_use_comet',
                        action='store_true',
                        help='use comet or not')
    parser.add_argument('--num_routing',
                        default=1,
                        type=int,
                        help='number of routing. Recommended: 0,1,2,3.')
    parser.add_argument('--dataset',
                        default='MovingMNist',
                        type=str,
                        help='dataset. so far only MovingMNist.')
    parser.add_argument('--backbone',
                        default='resnet',
                        type=str,
                        help='type of backbone. simple or resnet')
    parser.add_argument('--data_root',
                        default=('/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/'
                                 'data/MovingMNist'),
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
                        default=5e-4,
                        type=float,
                        help='weight decay')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--epoch', default=350, type=int, help='number of epoch')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='number of batch size')
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        help='adam or sgd.')
    parser.add_argument('--criterion',
                        default='triplet',
                        type=str,
                        help='triplet, nce, bce, or xent.')
    parser.add_argument('--nce_positive_frame_num',
                        default=10,
                        type=int,
                        help='the # of frames from anchor to use as positive.')
    parser.add_argument('--nce_temperature',
                        default=1.0,
                        type=float,
                        help='temperature to multiply with the similarity')
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

    # VideoClip info
    parser.add_argument('--num_videoframes', type=int, default=100)
    parser.add_argument('--dist_videoframes', type=int, default=50, help='the frame interval between each sequence.')
    parser.add_argument('--resize', type=int, default=128, help='to what to resize the frames of the video.')
    parser.add_argument('--fps', type=int, default=5, help='the fps for the loaded VideoClips')
    parser.add_argument('--count_videos', type=int, default=32, help='the number of video fiels to includuek')
    parser.add_argument('--count_clips', type=int, default=-1, help='the number of clips to includue')
    parser.add_argument(
        '--skip_videoframes',
        type=int,
        default=5,
        help='the number of video frames to skip in between each one. using 1 means that there is no skip.'
    )
    parser.add_argument('--gymnastics_video_directory',
                        default='/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/feb102020',
                        type=str,
                        help='video directory of gymnastics files.')

    args = parser.parse_args()
    args.use_comet = not args.no_use_comet
    assert args.num_routing > 0
    main(args)
