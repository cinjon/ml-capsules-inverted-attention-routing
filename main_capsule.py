#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All rights reserved.
#
'''Train CIFAR10 with PyTorch.'''
import argparse
from datetime import datetime
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import configs
from src.moving_mnist import MovingMNist
from src import capsule_model, diverse_multi_mnist
# from utils import progress_bar

# +
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
parser.add_argument(
    '--dataset',
    default='CIFAR10',
    type=str,
    help='dataset. CIFAR10, CIFAR100, DiverseMultiMNist, MovingMNist.')
parser.add_argument('--backbone',
                    default='resnet',
                    type=str,
                    help='type of backbone. simple or resnet')
parser.add_argument('--data_root',
                    default='/misc/kcgscratch1/ChoGroup/resnick/vidcaps',
                    type=str,
                    help='root of where to put the data.')
parser.add_argument('--num_workers',
                    default=2,
                    type=int,
                    help='number of workers. 0 or 2')
parser.add_argument('--num_gpus', default=1, type=int, help='number of gpus.')
parser.add_argument('--config',
                    default='resnet_backbone_cifar10',
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
                    help='learning rate. 0.1 for SGD')
parser.add_argument('--dp', default=0.0, type=float, help='dropout rate')
parser.add_argument('--weight_decay',
                    default=5e-4,
                    type=float,
                    help='weight decay')
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='random seed')
parser.add_argument('--epoch',
                    default=350,
                    type=int,
                    help="epoch")
parser.add_argument('--interactive',
                    action='store_true',
                    help='interactive mode')
# -

args = parser.parse_args()
assert args.num_routing > 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data
print('==> Preparing data..')
assert args.dataset in [
    'CIFAR10', 'CIFAR100', 'MovingMNist', 'DiverseMultiMNist'
]

config = getattr(configs, args.config).config

if 'CIFAR' in args.dataset:
    train_batch_size = int(args.num_gpus * 128 / 8)
    test_batch_size = int(args.num_gpus * 100 / 8)
    transform_train = config['transform_train']
    transform_test = config['transform_test']
    train_set = getattr(torchvision.datasets,
                        args.dataset)(root=args.data_root,
                                      train=True,
                                      download=True,
                                      transform=transform_train)
    test_set = getattr(torchvision.datasets,
                       args.dataset)(root=args.data_root,
                                     train=False,
                                     download=True,
                                     transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers)
    image_dim_size = 32
    loss_func = nn.CrossEntropyLoss()
    predicted_lambda = lambda v: v.max(dim=1)[1]
    correct_lambda = lambda predicted, targets: predicted.eq(targets).sum().item()
    total_lambda = lambda targets: targets.size(0)
    num_targets_lambda = lambda targets: targets.size(0)
elif args.dataset == 'MovingMNist':
    image_dim_size = 64
    train_batch_size = 16
    test_batch_size = 16

    # root = "/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules"
    train_set = MovingMNist(root=args.data_root, train=True)
    test_set = MovingMNist(root=args.data_root, train=False)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    loss_func = nn.BCEWithLogitsLoss()
    predicted_lambda = lambda v: (torch.sigmoid(v) > 0.5).type(v.dtype)
    correct_lambda = lambda predicted, targets:\
        predicted.eq(targets).all(1).sum().item()
    total_lambda = lambda targets: targets.size(0)
    num_targets_lambda = lambda targets: targets.eq(1).sum().item()
elif args.dataset == 'DiverseMultiMNist':
    train_batch_size = 128 # 5 * int(args.num_gpus * 128 / 8)
    test_batch_size = 128 # 5 * int(args.num_gpus * 128 / 8)
    image_dim_size = 36
    path = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps'
    train_set = diverse_multi_mnist.DiverseMultiMNist(
        path, train=True, download=True, batch_size=train_batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch_size, num_workers=args.num_workers)
    test_set = diverse_multi_mnist.DiverseMultiMNist(
        path, train=False, download=True, batch_size=test_batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, num_workers=args.num_workers)
    loss_func = nn.BCEWithLogitsLoss()
    predicted_lambda = lambda v: (torch.sigmoid(v) > 0.5).type(v.dtype)
    # The paper specifies that it is an IFF condition, i.e. predicted equals
    # targets iff they are the same everywhere.
    correct_lambda = lambda predicted, targets: predicted.eq(targets).all(1).sum().item()
    total_lambda = lambda targets: targets.size(0)
    num_targets_lambda = lambda targets: targets.eq(1).sum().item()

print('==> Building model..')
# Model parameters

print(config)
net = capsule_model.CapsModel(image_dim_size,
                              config['params'],
                              args.backbone,
                              args.dp,
                              args.num_routing,
                              sequential_routing=args.sequential_routing)

# +
optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=0.9,
                      weight_decay=args.weight_decay)

lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[150, 250],
                                                gamma=0.1)

# -


def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(net)
total_params = count_parameters(net)
print(total_params)

results_dir = os.path.join(args.data_root, "result") # '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results'
if not os.path.isdir(results_dir) and not args.debug:
    os.mkdir(results_dir)

today = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
store_dir = os.path.join(results_dir, today)
if not args.debug:
    os.mkdir(store_dir)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True


if args.resume_dir and not args.debug:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    true_correct = 0
    total = 0
    num_targets_total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # if batch_idx == len(train_loader):
        #     # This is dumb af.
        #     break
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        v = net(inputs)

        loss = loss_func(v, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = predicted_lambda(v)

        num_total = total_lambda(targets)
        num_targets = num_targets_lambda(targets)
        total += num_total
        num_targets_total += num_targets

        correct += correct_lambda(predicted, targets)
        true_positive_count = (predicted.eq(targets) & targets.eq(1)).sum().item()
        true_correct += true_positive_count

        if args.interactive:
            s = 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Tp: %.3f (%d / %d)'
            progress_bar(batch_idx, len(train_loader), s % (
                train_loss / (batch_idx + 1),
                100. * correct / total, correct, total,
                100. * true_correct / num_targets_total,
                true_correct,
                num_targets_total))
        else:
            print("Train Epoch:{} {}/{} Loss: {:.3f} | Acc: {:.3f}% ({}/{}) | Tp: {:.3f} ({}/{})".format(
                epoch, batch_idx+1, len(train_loader), train_loss / (batch_idx+1),
                100. * correct / total, correct, total,
                100. * true_correct / num_targets_total, true_correct,
                num_targets_total))

    return 100. * correct / total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    true_correct = 0
    total = 0
    num_targets_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if batch_idx == len(test_loader):
                # This is dumb af.
                break
            inputs = inputs.to(device)

            targets = targets.to(device)

            v = net(inputs)

            loss = loss_func(v, targets)

            test_loss += loss.item()

            predicted = predicted_lambda(v)

            num_total = total_lambda(targets)
            num_targets = num_targets_lambda(targets)
            total += num_total
            num_targets_total += num_targets

            correct += correct_lambda(predicted, targets)
            true_positive_count = (predicted.eq(targets) & targets.eq(1)).sum().item()
            true_correct += true_positive_count

            if args.interactive:
                s = 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Tp: %.3f (%d / %d)'
                progress_bar(batch_idx, len(test_loader), s % (
                    test_loss / (batch_idx + 1),
                    100. * correct / total, correct, total,
                    100. * true_correct / num_targets_total,
                    true_correct,
                    num_targets_total))
            else:
                print("Val Epoch:{} {}/{} Loss: {:.3f} | Acc: {:.3f}% ({}/{}) | Tp: {:.3f} ({}/{})".format(
                    epoch, batch_idx+1, len(test_loader), test_loss / (batch_idx+1),
                    100. * correct / total, correct, total,
                    100. * true_correct / num_targets_total, true_correct,
                    num_targets_total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc and not args.debug:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not args.debug:
            torch.save(state, os.path.join(store_dir, 'ckpt.pth'))
        best_acc = acc
    return 100. * correct / total


# +
results = {
    'total_params': total_params,
    'args': args,
    'params': config['params'],
    'train_acc': [],
    'test_acc': [],
}

total_epochs = args.epoch

if not args.debug:
    store_file = 'dataset_%s_num_routing_%s_backbone_%s.dct' % (
        str(args.dataset), str(args.num_routing), args.backbone)
    store_file = os.path.join(store_dir, store_file)

for epoch in range(start_epoch, start_epoch + total_epochs):
    results['train_acc'].append(train(epoch))

    lr_decay.step()
    results['test_acc'].append(test(epoch))
    if not args.debug:
        pickle.dump(results, open(store_file, 'wb'))
# -
