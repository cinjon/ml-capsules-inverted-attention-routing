"""
NOTE: This is just here to build in linpred_train while other models are trying
to launch.
"""
from datetime import datetime
import os
import getpass
import json
import time
import pickle
import random
import argparse
import socket

from comet_ml import Experiment as CometExperiment, OfflineExperiment
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import configs
import cinjon_point_jobs as cinjon_jobs
import linpred_train
import zeping_point_jobs as zeping_jobs
from src import capsule_points_model, capsule_ae
from src.shapenet import ShapeNet55
# from src.shapenet_original_dataset import Shapenet55Dataset
from src.modelnet import ModelNet


hostname = socket.gethostname()
is_prince = hostname.startswith('log-') or hostname.startswith('gpu-')


def run_tsne(model, path, epoch, args, comet_exp, num_classes):
    from MulticoreTSNE import MulticoreTSNE as multiTSNE

    train_loader, test_loader = get_loaders(args, rank=0, is_tsne=True)
    suffix = args.dataset

    model_poses = []
    model_presence = []
    targets = []

    with torch.no_grad():
        for loader, split in zip([train_loader, test_loader], ['train', 'test']):
            if split == 'test' and test_loader.dataset.split == 'train':
                continue

            # if split == 'train':
            #     continue

            for batch_num, (points, labels) in enumerate(loader):
                if batch_num % 10 == 0:
                    print('Batch %d / %d' % (batch_num, len(loader)))

                points = points.to('cuda')
                batch_size, num_points = points.shape[:2]

                # Change view so that we can put everything through the model at once.
                points = points.view(batch_size * num_points, *points.shape[2:])
                if args.criterion == 'autoencoder':
                    poses, presences = model(points, get_code=True)
                else:
                    poses, presences = model(points)
                poses = poses.view(batch_size, num_points, *poses.shape[1:])
                presences = presences.view(batch_size, num_points,
                                           *presences.shape[1:])
                poses = poses[:, 0]
                presences = presences[:, 0]
                poses = poses.cpu()
                presences = presences.cpu()
                points = points.cpu()
                model_poses.append(poses.view(batch_size, -1))
                model_presence.append(presences.view(batch_size, -1))
                targets.append(labels)

            model_poses = torch.cat(model_poses, 0)
            model_poses = model_poses.numpy()
            model_presence = torch.cat(model_presence, 0)
            model_presence = model_presence.numpy()
            targets = torch.cat(targets, 0)
            targets = targets.numpy().squeeze()

            for x, key in zip(
                    [model_poses, model_presence],
                    ['poses', 'presence']
            ):
                embeddings = multiTSNE(
                    n_components=2, perplexity=30, learning_rate=100.0
                ).fit_transform(x)

                vis_x = embeddings[:, 0]
                vis_y = embeddings[:, 1]
                plt.scatter(vis_x, vis_y, c=targets,
                            cmap=plt.cm.get_cmap("jet", num_classes), marker='.')
                plt.colorbar(ticks=range(num_classes))
                plt.clim(-0.5, float(num_classes) + 0.5)
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
            model_presence = []
            targets = []

    del model_presence
    del model_poses
    del targets
    del points
    del presences
    del poses


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


def run_modelnet_test(args, config, net, device, current_epoch, comet_exp=None):
    if 'pointcapsnet' in args.config:
        modelnet_net = capsule_ae.NewBackboneModel(config['params'], args, out_channels=40)
    elif 'resnet' in args.config:
        modelnet_net = capsule_points_model.BackboneModel(config['params'], args, out_channels=40)
    modelnet_net = modelnet_net.cuda()

    # Load weights from net
    model_dict = modelnet_net.state_dict()
    pretrained_dict = net.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if 'fc_head' not in k}
    model_dict.update(pretrained_dict)
    modelnet_net.load_state_dict(model_dict)

    # Freeze model
    for param in modelnet_net.parameters():
        param.requires_grad = False
    # Unfreeze last layer
    modelnet_net.fc_head.weight.requires_grad = True
    modelnet_net.fc_head.bias.requires_grad = True

    optimizer = optim.Adam(modelnet_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_set = ModelNet(args.modelnet_root, train=True)
    test_set = ModelNet(args.modelnet_root, train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True)

    # Train
    for epoch in range(args.modelnet_test_epoch):
        modelnet_net.train()

        averages = {
            'loss': Averager(),
            'grad_norm': Averager()
        }

        t = time.time()
        criterion = args.criterion
        optimizer.zero_grad()
        for batch_idx, (points, labels) in enumerate(train_loader):
            points = points[:, 0]
            points = points.cuda(device)
            labels = labels.squeeze()
            labels = labels.cuda(device)

            # output = net(points, return_embedding=True)
            # output = modelnet_net(output)
            output, _ = modelnet_net(points)
            loss = F.cross_entropy(output, labels)
            predictions = torch.argmax(output, dim=1)
            accuracy = (predictions == labels).float().mean().item()
            stats = {
                'accuracy': accuracy,
            }

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

            if batch_idx % 25 == 0 and batch_idx > 0:
                log_text = ('Train ModelNet Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
                print(log_text.format(epoch+1, batch_idx, len(train_loader),
                                      time.time() - t, averages['loss'].item())
                )
                t = time.time()

        train_loss = averages['loss'].item()
        train_acc = averages['accuracy'].item() if 'accuracy' in averages else None

        t = time.time()

        # Test
        modelnet_net.eval()

        averages = {
            'modelnet_%depoch_loss' % epoch: Averager()
        }

        t = time.time()

        with torch.no_grad():
            for batch_idx, (points, labels) in enumerate(test_loader):
                points = points[:, 0]
                points = points.cuda(device)
                labels = labels.squeeze()
                labels = labels.cuda(device)

                # output = net(points, return_embedding=True)
                # output = modelnet_net(output)
                output, _ = modelnet_net(points)
                loss = F.cross_entropy(output, labels)
                predictions = torch.argmax(output, dim=1)
                accuracy = (predictions == labels).float().mean().item()
                stats = {
                    'modelnet_%depoch_accuracy' % epoch: accuracy,
                }

                averages['modelnet_%depoch_loss' % epoch].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items() \
                                     if 'capsule_prob' not in k])

                if batch_idx % 100 == 0 and batch_idx > 0:
                    log_text = ('Val ModelNet {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
                    print(log_text.format(batch_idx, len(test_loader),
                                          time.time() - t, averages['modelnet_%depoch_loss' % epoch].item())
                    )
                    t = time.time()

        del points
        torch.cuda.empty_cache()

        extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                             for k, v in averages.items()])
        log_text = ('Val ModelNet {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
        print(log_text.format(batch_idx, len(test_loader),
                              time.time() - t, averages['modelnet_%depoch_loss' % epoch].item()))
        test_loss = averages['modelnet_%depoch_loss' % epoch].item()

        test_acc = averages['modelnet_%depoch_accuracy' % epoch].item() # if 'modelnet_accuracy' in averages else None

        if comet_exp:
            with comet_exp.test():
                epoch_avgs = {k: v.item() for k, v in averages.items()}
                comet_exp.log_metrics(epoch_avgs, epoch=current_epoch)


def get_loaders(args, rank=0, is_tsne=False):
    if 'shapenet' in args.dataset:
        stepsize_range = [float(k) for k in args.shapenet_stepsize_range.split(',')]
        stepsize_fixed = args.shapenet_stepsize_fixed
        if args.shapenet_rotation_train:
            rotation_train = [float(k) for k in args.shapenet_rotation_train.split(',')]
        else:
            rotation_train = None
        if args.shapenet_rotation_test:
            rotation_test = [float(k) for k in args.shapenet_rotation_test.split(',')]
        else:
            rotation_test = None
        rotation_same = args.shapenet_rotation_same
        root = os.path.join(args.data_root, args.dataset.replace('shapenet', 'dataset'))
        train_set = ShapeNet55(
            root, split='train', num_frames=args.num_frames,
            stepsize_fixed=stepsize_fixed, stepsize_range=stepsize_range,
            use_diff_object=args.use_diff_object,
            rotation_range=rotation_train, rotation_same=rotation_same)
        # NOTE: We changed this to train to test some shit.
        test_set = ShapeNet55(
            root, split='val', # 'train'
            num_frames=args.num_frames,
            stepsize_fixed=stepsize_fixed, stepsize_range=stepsize_range,
            use_diff_object=args.use_diff_object,
            rotation_range=rotation_test, rotation_same=rotation_same)

    if args.num_gpus == 1 or is_tsne:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
            # drop_last=True)
    else:
        print('Distributed dataloader', rank, args.num_gpus, args.batch_size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	    train_set,
    	    num_replicas=args.num_gpus,
    	    rank=rank)
        train_loader = torch.utils.data.DataLoader(
    	    dataset=train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True)

    # We should change this to shuffle=True dumbbutt.
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
        # drop_last=True)
    print('Returning data loaders...')
    return train_loader, test_loader


# Training
def train(epoch, step, net, optimizer, loader, args, device, comet_exp=None):
    net.train()

    if comet_exp:
        with comet_exp.train():
            comet_exp.log_current_epoch(epoch)

    averages = {
        'loss': Averager(),
        'grad_norm': Averager()
    }

    t = time.time()
    criterion = args.criterion
    optimizer.zero_grad()

    if args.num_gpus > 1:
        loader.sampler.set_epoch(epoch)

    for batch_idx, (points, labels) in enumerate(loader):
        if criterion == 'xent':
            points = points.cuda(device)
            batch_size, num_points = points.shape[:2]
            # Change view so that we can put everything through the model at once.
            points = points.view(batch_size * num_points, *points.shape[2:])
            labels = labels.squeeze()
            labels = labels.cuda(device)
            loss, stats = capsule_points_model.get_xent_loss(net, points, labels)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = 'Acc: {:.5f}.'.format(
                averages['accuracy'].item()
            )
        elif criterion == 'nceprobs_selective':
            points = points.cuda(device)
            loss, stats = capsule_points_model.get_nceprobs_selective_loss(
                net, points, device, epoch, args)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                 for k, v in averages.items() \
                                 if 'capsule_prob' not in k])
        elif criterion == 'backbone_nceprobs_selective':
            points = points.cuda(device)
            loss, stats = capsule_points_model.get_nceprobs_selective_loss(
                net, points, device, epoch, args)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                 for k, v in averages.items() \
                                 if 'capsule_prob' not in k])
        elif criterion == 'backbone_xent':
            points = points[:, 0]
            points = points.cuda(device)
            labels = labels.squeeze()
            labels = labels.cuda(device)
            loss, stats = capsule_points_model.get_backbone_test_loss(
                net, points, labels, args)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                 for k, v in averages.items() \
                                 if 'capsule_prob' not in k])
        elif criterion == 'autoencoder':
            points = points[:, 0]
            points = points.cuda(device)
            loss, stats = capsule_ae.get_autoencoder_loss(
                net, points, args)
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

        step += len(points) * args.num_gpus
        if comet_exp is not None:
            del points
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


def test(epoch, step, net, loader, args, device, store_dir=None, comet_exp=None):
    net.eval()

    averages = {
        'loss': Averager()
    }

    t = time.time()
    criterion = args.criterion

    with torch.no_grad():
        for batch_idx, (points, labels) in enumerate(loader):
            if criterion == 'xent':
                points = points.cuda(device)
                batch_size, num_points = points.shape[:2]
                # Change view so that we can put everything through the model at once.
                points = points.view(batch_size * num_points, *points.shape[2:])
                labels = labels.squeeze()
                labels = labels.cuda(device)
                loss, stats = capsule_points_model.get_xent_loss(net, points, labels)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = 'Acc: {:.5f}.'.format(
                    averages['accuracy'].item()
                )
            elif criterion == 'nceprobs_selective':
                points = points.cuda(device)
                loss, stats = capsule_points_model.get_nceprobs_selective_loss(
                    net, points, device, epoch, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items() \
                                     if 'capsule_prob' not in k])
            elif criterion == 'backbone_nceprobs_selective':
                points = points.cuda(device)
                loss, stats = capsule_points_model.get_nceprobs_selective_loss(
                    net, points, device, epoch, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items() \
                                     if 'capsule_prob' not in k])
            elif criterion == 'backbone_xent':
                points = points[:, 0]
                points = points.cuda(device)
                labels = labels.squeeze()
                labels = labels.cuda(device)
                loss, stats = capsule_points_model.get_backbone_test_loss(
                    net, points, labels, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items() \
                                     if 'capsule_prob' not in k])
            elif criterion == 'autoencoder':
                points = points[:, 0]
                points = points.cuda(device)
                # labels = labels.squeeze()
                # labels = labels.cuda(device)
                loss, stats = capsule_ae.get_autoencoder_loss(
                    net, points, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    if key not in averages:
                        averages[key] = Averager()
                    averages[key].add(value)
                extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                                     for k, v in averages.items() \
                                     if 'capsule_prob' not in k])

            if batch_idx % 100 == 0 and batch_idx > 0:
                log_text = ('Val Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
                print(log_text.format(epoch + 1, batch_idx, len(loader),
                                      time.time() - t, averages['loss'].item())
                )
                t = time.time()

    del points
    torch.cuda.empty_cache()

    extra_s = ', '.join(['{}: {:.5f}.'.format(k, v.item())
                         for k, v in averages.items()])
    log_text = ('Val Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
    print(log_text.format(epoch + 1, batch_idx, len(loader),
                          time.time() - t, averages['loss'].item()))
    test_loss = averages['loss'].item()

    if comet_exp:
        with comet_exp.test():
            epoch_avgs = {k: v.item() for k, v in averages.items()}
            comet_exp.log_metrics(epoch_avgs, step=step, epoch=epoch)

    test_acc = averages['accuracy'].item() if 'accuracy' in averages else None
    return test_loss, test_acc


def main(gpu, args, port=12355, initialize=True):
    if initialize:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend='nccl', rank=gpu, world_size=args.num_gpus)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = getattr(configs, args.config).config
    num_classes = args.num_output_classes
    print('Num Classes: %d' % num_classes)
    print(config)

    train_loader, test_loader = get_loaders(args, rank=gpu)
    num_frames = 3

    linear_classifier_out = args.num_output_classes if args.criterion == 'xent' else None

    print('==> Building model..')
    if 'backbone' in args.criterion:
        if 'pointcapsnet' in args.config:
            net = capsule_ae.NewBackboneModel(config['params'], args)
        elif 'resnet' in args.config:
            net = capsule_points_model.BackboneModel(config['params'], args)
    elif args.criterion == 'autoencoder':
        net = capsule_ae.PointCapsNet(config['params'], args)
    else:
        if 'pointcapsnet' in args.config:
            net = capsule_ae.NewBackboneModel(config['params'], args)
        elif 'resnet' in args.config:
            net = capsule_points_model.CapsulePointsModel(
                config['params'], args, linear_classifier_out)
    print(net)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_scheduler:
        # CIFAR used milestones of [150, 250] with gamma of 0.1
        milestones = [int(k) for k in args.schedule_milestones.split(',')]
        gamma = args.schedule_gamma
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=milestones,
                                                         gamma=gamma)

    total_params = count_parameters(net)
    print('Total Params %d' % total_params)

    today = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    if args.linpred_test_only:
        store_dir = None
    else:
        assert(args.name is not None and args.counter is not None)
        store_dir = os.path.join(args.results_dir, args.name, str(args.counter), today)

    if not args.linpred_test_only and gpu == 0 and not os.path.isdir(store_dir) and not args.debug:
        os.makedirs(store_dir)

    torch.cuda.set_device(gpu)
    net.cuda(gpu)
    net = DDP(net, device_ids=[gpu], find_unused_parameters=True)

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if args.resume_dir and not args.debug:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        try:
            checkpoint = torch.load(os.path.join(
                args.resume_dir, 'ckpt.epoch%d.pth' % args.resume_epoch))
        except Exception as e:
            checkpoint = torch.load(os.path.join(
                args.resume_dir, 'ckpt.epoch%d.best.pth' % args.resume_epoch))
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    results = {
        'total_params': total_params,
        'args': args,
        'params': config['params'],
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
    }

    total_epochs = args.epoch

    comet_exp = None
    if args.use_comet and gpu == 0:
        comet_exp = CometExperiment(api_key="hIXq6lDzWzz24zgKv7RYz6blo",
                                    project_name="capsules-shapenet-filtered",
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

    # torch.autograd.set_detect_anomaly(True)

    step = 0
    last_test_loss = 1e8
    last_saved_epoch = 0
    device = gpu

    if args.linpred_test_only:
        print('\n***\nStarting LinPred Test on ShapeNet (%d)\n***' % start_epoch)
        linpred_train.run_ssl_shapenet(start_epoch, net, args, config, comet_exp)
        print('\n***\nEnded LinPred Test on ShapeNet (%d)\n***' % start_epoch)
        print('\n***\nStarting LinPred Test on ModelNet (%d)\n***' % start_epoch)
        linpred_train.run_ssl_modelnet(start_epoch, net, args, config, comet_exp)
        print('\n***\nEnded LinPred Test on ModelNet (%d)\n***' % start_epoch)

        if args.counter in [62, 63, 66, 67]:
            if args.resume_epoch < 66:
                args.resume_epoch += 6
                main(gpu, args, initialize=False)
            elif args.classifier_type == 'pose':
                args.classifier_type = 'presence'
                args.resume_epoch = 6
                main(gpu, args, initialize=False)
        elif args.counter in [68]:
            if args.classifier_type == 'pose':
                args.classifier_type = 'presence'
                main(gpu, args, initialize=False)
        return

    if args.linpred_svm_only:
        print('\n***\nStarting LinPred SVM on ShapeNet (%d)\n***' % start_epoch)
        linpred_train.run_svm_shapenet(start_epoch, net, args, config, comet_exp)
        print('\n***\nEnded LinPred SVM on ShapeNet (%d)\n***' % start_epoch)
        return

    for epoch in range(start_epoch, start_epoch + total_epochs):
        print('Starting Epoch %d' % epoch)
        train_loss, train_acc, step = train(epoch, step, net, optimizer,
                                            train_loader, args, device, comet_exp)

        if scheduler:
            scheduler.step()

        if epoch % 3 == 0 and gpu == 0:
            test_loss, test_acc = test(
                epoch, step, net, test_loader, args, device,
                store_dir=store_dir, comet_exp=comet_exp)

            if not args.debug and epoch >= last_saved_epoch + 4 and \
               last_test_loss > test_loss:
                state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'loss': test_loss
                }
                print('\n***\nSaving best epoch %d\n***\n' % epoch)
                torch.save(state, os.path.join(store_dir, 'ckpt.epoch%d.best.pth' % epoch))
                last_test_loss = test_loss
                last_saved_epoch = epoch

            if not args.debug and epoch >= last_saved_epoch + 5 and \
               args.epoch >= 100 and args.criterion == 'nceprobs_selective':
                state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'loss': test_loss
                }
                print('\n***\nSaving epoch %d\n***\n' % epoch)
                torch.save(state, os.path.join(store_dir, 'ckpt.epoch%d.pth' % epoch))
                last_saved_epoch = epoch

        if comet_exp:
            comet_exp.log_epoch_end(epoch)

        if all([
            epoch % args.do_svm_shapenet_every == 0,
            epoch > args.do_svm_shapenet_after,
        ]):
            print('\n***\nStarting SVM ShapeNet Test (%d)\n***' % epoch)
            linpred_train.run_svm_shapenet(epoch, net, args, config, comet_exp)

        if all([
                gpu == 0,
                args.do_tsne_test_every is not None,
                epoch % args.do_tsne_test_every == 0,
                epoch > args.do_tsne_test_after
        ]):
            print('\n***\nStarting TSNE (%d)\n***' % epoch)
            # NOTE: We always put center_start as true because we are using this
            # as a test on the digits. That's what hte linear separation is about.
            # (the angle doesn't matter).
            run_tsne(net, store_dir, epoch, args, comet_exp, num_classes)
            print('\n***\nEnded TSNE (%d)\n***' % epoch)
            torch.cuda.empty_cache()
            print('\n***\nCleared Cache (%d)\n***' % epoch)

        if all([
                args.do_modelnet_test_every is not None,
                epoch % args.do_modelnet_test_every == 0,
                epoch > args.do_modelnet_test_after,
        ]):
            print('\n***\nStarting ModelNet Test (%d)\n***' % epoch)
            run_modelnet_test(args, config, net, device, epoch, comet_exp=comet_exp)

        # if all([
        #     epoch % args.do_svm_shapenet_every == 0,
        #     epoch > args.do_svm_shapenet_after,
        # ]):
        #     print('\n***\nStarting SVM ShapeNet Test (%d)\n***' % epoch)
        #     linpred_train.run_svm_shapenet(epoch, net, args, config, comet_exp)


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
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='the checkpoint epoch to resume at')
    parser.add_argument('--no_use_comet',
                        action='store_true',
                        help='use comet or not')
    parser.add_argument('--linpred_test_only', action='store_true',
                        help='whether to only do the linpred test and exit')
    parser.add_argument('--linpred_test_class_balanced', action='store_true',
                        help='whether we use class balanced sampling in linpred')
    parser.add_argument('--linpred_svm_only', action='store_true',
                        help='whether to only do the linpred test and exit')
    parser.add_argument('--do_tsne_test_every',
                        default=None, type=int,
                        help='if an integer > 0, then do the mnist_affnist test ' \
                        'every this many epochs.')
    parser.add_argument('--do_tsne_test_after',
                        default=10, type=int,
                        help='if an integer > 0, then do the mnist_affnist test ' \
                        'starting at this epoch.')
    parser.add_argument('--num_routing',
                        default=1,
                        type=int,
                        help='number of routing. Recommended: 0,1,2,3.')
    parser.add_argument('--dataset',
                        default='shapenet5',
                        type=str,
                        help='dataset: either shapenet5, shapenet16 or shapenetFull.')
    parser.add_argument('--num_frames', default=2, type=int,
                        help='how many frames to include in the input.')
    parser.add_argument('--use_diff_object',
                        action='store_true',
                        help='whether to use the class sampler with MovingMNist2')
    parser.add_argument('--shapenet_stepsize_range', type=str, default='1,1',
                        help='comma separated 2-tuple of the stepsize bounds.')
    parser.add_argument('--shapenet_stepsize_fixed', default=None, type=float,
                        help='if fixing the step size between frames.')
    parser.add_argument('--shapenet_rotation_train', type=str, default='-180,180',
                        help='if given, is a comma separated 2-tuple of the allowed rotation degrees for train.')
    parser.add_argument('--shapenet_rotation_test', type=str, default='-180,180',
                        help='if given, is a comma separated 2-tuple of the allowed rotation degrees for test.')
    parser.add_argument('--shapenet_rotation_same', action='store_true',
                        help='whether we rotate the objects the same amount.')
    parser.add_argument('--data_root',
                        default='/misc/kcgscratch1/ChoGroup/resnick/vidcaps/shapenet',
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
    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help='learning rate. 0.1 for SGD, 1e-2 for Adam (trying)')
    parser.add_argument('--weight_decay',
                        default=0, # was 5e-4 but default off
                        type=float,
                        help='weight decay')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--epoch', default=350, type=int, help='number of epoch')
    parser.add_argument('--linear_batch_size',
                        default=42,
                        type=int,
                        help='number of batch size')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='number of batch size')
    parser.add_argument('--optimizer',
                        default='adam',
                        type=str,
                        help='adam or sgd.')
    parser.add_argument('--criterion',
                        default='xent',
                        type=str)
    parser.add_argument('--nce_presence_temperature',
                        default=1.0,
                        type=float,
                        help='temperature to multiply with the similarity')
    parser.add_argument('--nce_presence_lambda',
                        default=1.0,
                        type=float,
                        help='lambda on the nce loss.')
    parser.add_argument('--presence_type', default='l2norm', type=str)
    parser.add_argument('--use_scheduler',
                        action='store_true',
                        help='whether to use the scheduler or not.')
    parser.add_argument('--schedule_milestones', type=str, default='150,250',
                        help='the milestones in the LR.')
    parser.add_argument('--schedule_gamma', type=float, default=0.1,
                        help='the default LR gamma.')

    # Presence
    parser.add_argument('--simclr_selection_strategy',
                        type=str,
                        default='default',
                        help='type of selection strategy for simclr nceprobs.')
    parser.add_argument('--num_output_classes',
                        default=55, # coudl be 5 though.
                        type=float,
                        help='the number of classes. this is a hack and dumb.')

    parser.add_argument('--classifier_type',
                        default='presence', # presence or pose.
                        type=str)

    parser.add_argument('--dynamic_routing',
                        action='store_true',
                        help='whether to use dynamic routing in NewBackboneModel')

    # ModelNet
    parser.add_argument('--modelnet_root',
                        type=str,
                        default='/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/modelnet/modelnet40_ply_hdf5_2048',
                        help='path to modelnet data')
    parser.add_argument('--do_modelnet_test_every',
                        default=25, type=int,
                        help='if an integer > 0, then do the mnist_affnist test ' \
                        'every this many epochs.')
    parser.add_argument('--do_modelnet_test_after',
                        default=199, type=int,
                        help='if an integer > 0, then do the mnist_affnist test ' \
                        'starting at this epoch.')
    parser.add_argument('--modelnet_test_epoch',
                        default=10, type=int,
                        help='ModelNet test epoch number')

    # SVM
    parser.add_argument('--do_svm_shapenet_every',
                        default=25, type=int,
                        help='if an integer > 0, then do the svm shapenet test ' \
                        'every this many epochs.')
    parser.add_argument('--do_svm_shapenet_after',
                        default=199, type=int,
                        help='if an integer > 0, then do the svm shapenet test ' \
                        'starting at this epoch.')


    args = parser.parse_args()
    if args.mode == 'jobarray':
        jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        user = getpass.getuser()
        user = 'cinjon' if user in ['cr2668', 'resnick'] else 'zeping'
        if user == 'zeping':
            counter, job = zeping_jobs.run(find_counter=jobid)
        elif user == 'cinjon':
            counter, job = cinjon_jobs.run(find_counter=jobid)
        else:
            raise

        for key, value in job.items():
            setattr(args, key, value)
        print(counter, job, '\n', args, '\n')

    args.use_comet = (not args.no_use_comet) and (not args.debug)
    assert args.num_routing > 0

    if user == 'cinjon':
        if args.counter == 34:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.15/34/2020-06-15-05-54-27')
            args.resume_epoch = 150
            args.classifier_type = 'pose'
        elif args.counter == 50:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.16/50/2020-06-16-13-30-09')
            # TODO: Also do resume_epoch = 66 with this one.
            args.resume_epoch = 51
            args.classifier_type = 'pose'
        elif args.counter == 51:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.16/51/2020-06-16-13-32-26')
            args.classifier_type = 'pose'
            args.resume_epoch = 51
        elif args.counter == 52:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.16/52/2020-06-16-17-42-40/')
            args.resume_epoch = 36
            args.classifier_type = 'pose'
        elif args.counter == 53:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.16/53/2020-06-16-17-41-58')
            args.resume_epoch = 45
            args.classifier_type = 'pose'
        elif args.counter == 54:
            args.linpred_test_only = True
            args.linpred_test_class_balanced = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.16/54/2020-06-16-17-41-57/')
            args.resume_epoch = 42
            args.classifier_type = 'pose'
        elif args.counter == 55:
            args.linpred_test_only = True
            args.linpred_test_class_balanced = True
            # args.linpred_svm_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.16/55/2020-06-16-17-41-56/')
            args.resume_epoch = 36
            args.classifier_type = 'pose'
        elif args.counter == 56:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.17/56/2020-06-17-09-19-39')
            # args.resume_epoch = 78
            args.resume_epoch = 60
            # args.classifier_type = 'presence'
            # NOTE: DONE THIS ONE
            args.classifier_type = 'pose'
        elif args.counter == 57:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.17/57/2020-06-17-09-19-39/')
            # args.resume_epoch = 78
            args.resume_epoch = 60
            args.classifier_type = 'pose'
        elif args.counter == 58:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.17/58/2020-06-17-09-19-40/')
            # args.resume_epoch = 78
            args.resume_epoch = 60
            args.classifier_type = 'pose'
        elif args.counter == 59:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.17/59/2020-06-17-09-19-40/')
            # args.resume_epoch = 78
            args.resume_epoch = 60
            args.classifier_type = 'pose'
        elif args.counter == 60:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.18/60/2020-06-18-16-10-08')
            # args.resume_epoch = 78
            args.resume_epoch = 54
            args.classifier_type = 'pose'
        elif args.counter == 61:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.18/61/2020-06-18-16-10-08/')
            # args.resume_epoch = 78
            args.resume_epoch = 60
            args.classifier_type = 'pose'
        elif args.counter == 62:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.18/62/2020-06-18-16-23-17')
            args.resume_epoch = 12
            args.classifier_type = 'pose'
        elif args.counter == 63:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.18/63/2020-06-18-16-23-22')
            args.resume_epoch = 12
            args.classifier_type = 'pose'
        elif args.counter == 66:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.20/66/2020-06-20-13-30-08')
            args.resume_epoch = 12
            args.classifier_type = 'pose'
        elif args.counter == 67:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.20/67/2020-06-20-13-30-25')
            args.resume_epoch = 12
            args.classifier_type = 'pose'
        elif args.counter == 68:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.21/68/2020-06-21-11-57-37')
            args.resume_epoch = 60
            args.classifier_type = 'pose'
        elif args.counter == 69:
            args.linpred_test_only = True
            args.num_gpus = 1
            base_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results/shapenet'
            args.resume_dir = os.path.join(
                base_dir, '2020.06.20/67/2020-06-20-13-30-25')
            args.resume_epoch = 12
            args.classifier_type = 'pose'

    default_port = random.randint(10000, 19000)
    mp.spawn(main, nprocs=args.num_gpus, args=(args, default_port))
