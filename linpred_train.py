"""

Sample Command:
python linpred_train.py --criterion xent --resume_dir /.../resnick/vidcaps/results/cfgmnist-xent-nroute1/2020-03-27-10-49-05 \
--debug --data_root /.../resnick/vidcaps --batch_size 32 --num_routing 1
--dataset affnist --test_only --debug --checkpoint_epoch 2 --config resnet_backbone_mnist
"""
import copy
import os
import json
import time
import pickle
import random
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import configs
from src import capsule_time_model
from src.moving_mnist.moving_mnist2 import MovingMNIST as MovingMNist2
from src.affnist import AffNist


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def get_loaders(args=None, data_root=None, batch_size=None):
    batch_size = batch_size or args.batch_size
    data_root = data_root or args.data_root
    mnist_root = os.path.join(data_root, 'mnist')
    affnist_root = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/affnist'
    mnist_train_transforms = [
        transforms.Pad(12),
        transforms.RandomCrop(40),
        # transforms.Resize(args.resize_data),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    mnist_test_transforms = [
        transforms.Pad(6),
        # transforms.Resize(args.resize_data),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

    train_set = torchvision.datasets.MNIST(
        mnist_root, train=True, download=True,
        transform=transforms.Compose(mnist_train_transforms)
    )
    affnist_test_set = AffNist(
        affnist_root, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    test_set = torchvision.datasets.MNIST(
        mnist_root, train=False,
        transform=transforms.Compose(mnist_test_transforms)
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    affnist_loader = torch.utils.data.DataLoader(
        affnist_test_set, batch_size=192, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=192, shuffle=False)
    return train_loader, test_loader, affnist_loader


def run_ssl_model(ssl_epoch, model, mnist_root, affnist_root, comet_exp, batch_size,
                  colored, num_workers, config_params, backbone, num_routing,
                  num_frames, lr, weight_decay, affnist_subset):
    train_set = MovingMNist2(mnist_root, train=True, seq_len=1, image_size=64,
                             colored=colored, tiny=False, num_digits=1)
    test_set = MovingMNist2(mnist_root, train=False, seq_len=1, image_size=64,
                            colored=colored, tiny=False, num_digits=1)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)
    # affnist_root = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/affnist'
    affnist_test_set = AffNist(
        affnist_root, train=False, subset=affnist_subset,
        transform=transforms.Compose([
            transforms.Pad(12),
            transforms.ToTensor(),
        ])
    )
    affnist_test_loader = torch.utils.data.DataLoader(
        affnist_test_set, batch_size=batch_size, shuffle=False)

    dp = 0.0
    net = capsule_time_model.CapsTimeModel(config_params,
                                           backbone,
                                           dp,
                                           num_routing,
                                           sequential_routing=False,
                                           num_frames=num_frames,
                                           mnist_classifier_head=True)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.to('cuda')
    net = torch.nn.DataParallel(net)

    current_state_dict = net.state_dict()
    trained_state_dict = model.state_dict()
    current_state_dict.update(trained_state_dict)
    net.load_state_dict(current_state_dict)
    for name, param in net.named_parameters():
        if 'mnist_classifier' not in name:
            param.requires_grad = False

    start_epoch = 0
    total_epochs = 20
    for epoch in range(start_epoch, start_epoch + total_epochs):
        train_loss, train_acc = train(epoch, net, optimizer, train_loader)
        if comet_exp is not None:
            with comet_exp.train():
                comet_exp.log_metrics(
                    {'ssl%d acc/mnist' % ssl_epoch: train_acc,
                     'ssl%d loss/mnist' % ssl_epoch: train_loss},
                    step=epoch
                )

        test_loss, test_acc = test(epoch, net, test_loader)
        test_metrics = {
            'ssl%d acc/mnist' % ssl_epoch: test_acc,
            'ssl%d loss/mnist' % ssl_epoch: test_loss
        }
        loss_str = 'Train Loss %.6f, Test Loss %.6f' % (
            train_loss, test_loss
        )
        acc_str = 'Train Acc %.6f, Test Acc %.6f' % (
            train_acc, test_acc
        )

        if test_acc > .9:
            affnist_loss, affnist_acc = test(epoch, net, affnist_test_loader)
            test_metrics.update({
                'ssl%d acc/affnist' % ssl_epoch: affnist_acc,
                'ssl%d loss/affnist' % ssl_epoch: affnist_loss
            })
            loss_str += ', Affnist Loss %.6f' % affnist_loss
            acc_str += ', Affnist Acc %.6f' % affnist_acc

        if comet_exp is not None:
            with comet_exp.test():
                comet_exp.log_metrics(test_metrics, step=epoch)

        print('Epoch %d:\n\t%s\n\t%s' % (epoch, loss_str, acc_str))


def get_mnist_loss(model, images, labels):
    batch_size, num_images = images.shape[:2]

    # Change view so that we can put everything through the model at once.
    images = images.view(batch_size * num_images, *images.shape[2:])
    output, poses = model(images, return_mnist_head=True)
    labels = labels.squeeze()
    loss = F.cross_entropy(output, labels)
    predictions = torch.argmax(output, dim=1)
    accuracy = (predictions == labels).float().mean().item()
    stats = {
        'accuracy': accuracy,
    }
    return loss, stats


# Training
def train(epoch, net, optimizer, loader):
    net.train()

    averages = {
        'loss': Averager()
    }

    t = time.time()
    optimizer.zero_grad()
    device = 'cuda'
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        loss, stats = get_mnist_loss(net, images, labels)
        averages['loss'].add(loss.item())
        for key, value in stats.items():
            if key not in averages:
                averages[key] = Averager()
            averages[key].add(value)
        extra_s = 'Acc: {:.5f}.'.format(
            averages['accuracy'].item()
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 100 == 0:
            log_text = ('Train Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
            print(log_text.format(epoch+1, batch_idx, len(loader),
                                  time.time() - t, averages['loss'].item())
            )
            t = time.time()

    train_loss = averages['loss'].item()
    train_acc = averages['accuracy'].item()
    return train_loss, train_acc


def test(epoch, net, loader):
    net.eval()
    averages = {
        'loss': Averager()
    }

    t = time.time()
    device = 'cuda'
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            loss, stats = get_mnist_loss(net, images, labels)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                if key not in averages:
                    averages[key] = Averager()
                averages[key].add(value)
            extra_s = 'Acc: {:.5f}.'.format(
                averages['accuracy'].item()
            )

            if batch_idx % 100 == 0:
                log_text = ('Val Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
                print(log_text.format(epoch + 1, batch_idx, len(loader),
                                      time.time() - t, averages['loss'].item())
                )
                t = time.time()

        test_loss = averages['loss'].item()
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

    train_loader, test_loader, affnist_loader = get_loaders(args)

    print('==> Building model..')
    sequential_routing = args.sequential_routing
    if args.test_only:
        # Use this for models that already trained with discrimative loss.
        print('Building with NO mnist classifier head.')
        net = capsule_model.CapsModel(config['params'],
                                      args.backbone,
                                      args.dp,
                                      args.num_routing,
                                      sequential_routing=sequential_routing)
    elif args.dataset in ['mnist', 'affnist']:
        print('Building with YES mnist classifier head.')
        # Use this for models that need to a linear classifier trained on top.
        net = capsule_model.CapsModel(config['params'],
                                      args.backbone,
                                      args.dp,
                                      args.num_routing,
                                      sequential_routing=sequential_routing,
                                      mnist_classifier_head=True)

    if args.test_only:
        optimizer = None
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=args.lr,
                               # weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(),
                              lr=args.lr,
                              momentum=0.9,
                              weight_decay=args.weight_decay)

    today = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    if args.test_only:
        store_dir = os.path.join(
            args.resume_dir,
            'test-%s' % args.dataset,
            'ckpt%d' % args.checkpoint_epoch
        )
    else:
        store_dir = os.path.join(
            args.resume_dir,
            'linpred-%s' % args.dataset,
            'ckpt%d' % args.checkpoint_epoch
        )

    if not os.path.isdir(store_dir) and not args.debug:
        os.makedirs(store_dir)

    net = net.to(device)
    if device == 'cuda':
        if args.num_gpus > 1:
            net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(
        os.path.join(args.resume_dir, 'ckpt.epoch%d.pth' % args.checkpoint_epoch)
    )
    state_dict = checkpoint['net']
    if args.test_only:
        print('Loading from state dict ...')
        print(net.pre_caps.pre_caps[1].bias)
        net.load_state_dict(state_dict)
        print(net.pre_caps.pre_caps[1].bias)
    else:
        curr_state_dict = net.state_dict()
        curr_state_dict.update(state_dict)
        net.load_state_dict(curr_state_dict)
        for name, param in net.named_parameters():
            if 'mnist_classifier' not in name:
                param.requires_grad = False

    results = {
        'args': args,
        'params': config['params'],
        'train_loss': [],
        'test_loss': [],
    }

    total_epochs = args.epoch
    if not args.debug:
        store_file = 'dataset_%s_num_routing_%s_backbone_%s.dct' % (
            str(args.dataset), str(args.num_routing), args.backbone
        )
        store_file = os.path.join(store_dir, store_file)

    if args.test_only:
        test_loss, test_acc = test(0, net, args.criterion, test_loader, args, device, store_dir=store_dir)
        print('Test Acc %.4f / Loss %.4f.' % (test_acc, test_loss))
        results['test_loss'].append(test_loss)

        affnist_test_loss, affnist_test_acc = test(0, net, args.criterion, affnist_loader, args, device, store_dir=store_dir)
        print('Affnist Test Acc %.4f / Loss %.4f.' % (affnist_test_acc, affnist_test_loss))

        if not args.debug:
            with open(store_file, 'wb') as f:
                pickle.dump(results, f)

        return

    for epoch in range(start_epoch, start_epoch + total_epochs):
        train_loss, train_acc = train(epoch, net, optimizer, args.criterion, train_loader, args, device)
        results['train_loss'].append(train_loss)

        # if scheduler:
        #     scheduler.step()

        test_loss, test_acc = test(epoch, net, args.criterion, test_loader, args, device, store_dir=store_dir)
        results['test_loss'].append(test_loss)

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
    parser.add_argument('--num_routing',
                        default=1,
                        type=int,
                        help='number of routing. Recommended: 0,1,2,3.')
    parser.add_argument('--dataset',
                        default='mnist',
                        type=str,
                        help='mnist or affnist.')
    parser.add_argument('--backbone',
                        default='resnet',
                        type=str,
                        help='type of backbone. simple or resnet')
    parser.add_argument('--data_root',
                        default=('/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/'
                                 'data/MovingMNist'),
                        type=str,
                        help='root of where to put the data.')
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
    parser.add_argument('--test_only',
                        action='store_true',
                        help='whether we are only testing and not training. e.g. testing mnist discrimative on affnist.')
    parser.add_argument('--checkpoint_epoch', type=int, help='which epoch to use.')
    parser.add_argument('--resize_data', type=int, default=40,
                        help='to what to resize the data.')

    args = parser.parse_args()
    assert args.num_routing > 0
    main(args)
