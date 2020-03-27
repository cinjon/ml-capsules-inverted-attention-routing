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
from torchvision.datasets import MNIST

import configs
from src.affnist.dataset import AffNist
from src import capsule_model

# +
parser = argparse.ArgumentParser(
    description='Training Capsules using Inverted Dot-Product Attention Routing'
)

parser.add_argument(
    '--resume_dir',
    '-r',
    default='',
    type=str,
    help='dir where we resume from checkpoint')
parser.add_argument(
    '--num_routing',
    default=3,
    type=int,
    help='number of routing. Recommended: 0,1,2,3.')
parser.add_argument(
    '--dataset',
    default='MNist',
    type=str,
    help='dataset. MovingMNist, MNist.')
parser.add_argument(
    '--backbone',
    default='resnet',
    type=str,
    help='type of backbone. simple or resnet')
# parser.add_argument(
#     '--data_root',
#     default='/misc/kcgscratch1/ChoGroup/resnick/vidcaps',
#     type=str,
#     help='root of where to put the data.')
parser.add_argument(
    '--num_workers',
    default=2,
    type=int,
    help='number of workers. 0 or 2')
parser.add_argument(
    '--config',
    default='resnet_backbone_affnist',
    type=str,
    help='path of the config')
parser.add_argument(
    '--debug',
    action='store_true',
    help='use debug mode (without saving to a directory)')
parser.add_argument(
    '--sequential_routing',
    action='store_true',
    help='not using concurrent_routing')
parser.add_argument(
    '--lr',
    default=0.1,
    type=float,
    help='learning rate. 0.1 for SGD')
parser.add_argument(
    '--dp', default=0.0, type=float, help='dropout rate')
parser.add_argument(
    '--weight_decay',
    default=5e-4,
    type=float,
    help='weight decay')
parser.add_argument(
    '--seed',
    default=0,
    type=int,
    help='random seed')
parser.add_argument(
    '--epoch',
    default=350,
    type=int,
    help="epoch")
parser.add_argument(
    '--interactive',
    action='store_true',
    help='interactive mode')
# -

args = parser.parse_args()
assert args.num_routing > 0

if args.interactive:
    from utils import progress_bar

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
# assert args.dataset in [
#     'CIFAR10', 'CIFAR100', 'MovingMNist', 'DiverseMultiMNist', 'MNist', 'affNIST'
# ]
assert args.dataset in ['MNist', 'MovingMNist']

config = getattr(configs, args.config).config

path = '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/data'
if args.dataset == 'MNist':
    train_batch_size = 64
    test_batch_size = 64
    image_dim_size = 40
    train_set = MNIST(
        path,
        train=True,
        download=True,
        transform=transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Pad(12),
            transforms.RandomCrop(40),
            transforms.ToTensor(),
            transforms.Normalize((0.1397,), (0.3081,))]))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    test_set = MNIST(
        path,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Pad(6),
            transforms.ToTensor(),
            transforms.Normalize((0.1397,), (0.3081,))]))
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    loss_func = nn.CrossEntropyLoss()
    predicted_lambda = lambda v: v.max(dim=1)[1]
    correct_lambda = lambda predicted, targets: predicted.eq(targets).sum().item()
    total_lambda = lambda targets: targets.size(0)
    num_targets_lambda = lambda targets: targets.size(0)
elif args.dataset == 'MovingMNist':
    train_batch_size = 16
    test_batch_size = 16
    image_dim_size = 64
    train_set = MovingMNist(
        root=os.path.join(path, 'MovingMNist'), train=True)
    test_set = MovingMNist(
        root=os.path.join(path, 'MovingMNist'), train=False)
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

# affnist_train_set = AffNist(affnist_path, train=True, subset=True)
# afnist_train_loader = torch.utils.data.DataLoader(
#     dataset=affnist_train_set,
#     batch_size=affnist_train_batch_size,
#     shuffle=True,
#     num_workers=args.num_workers)
affnist_test_set = AffNist(
    os.path.join(path, 'affNIST'),
    train=False,
    subset=True,
    transform=transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Pad(12),
        transforms.ToTensor(),
        transforms.Normalize((0.1397,), (0.3081,))]))
affnist_test_loader = torch.utils.data.DataLoader(
    dataset=affnist_test_set,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=args.num_workers)
affnist_loss_func = nn.CrossEntropyLoss()
affnist_predicted_lambda = lambda v: v.max(dim=1)[1]
affnist_correct_lambda = lambda predicted, targets: predicted.eq(targets).sum().item()
affnist_total_lambda = lambda targets: targets.size(0)
affnist_num_targets_lambda = lambda targets: targets.size(0)

print('==> Building model..')
# Model parameters

print(config)
net = capsule_model.CapsModel(config['params'],
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

# results_dir = '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results'
results_dir = '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/result'
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
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        v = net.forward(inputs, return_embedding=False)

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
    # global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    true_correct = 0
    total = 0
    num_targets_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            v = net.forward(inputs, return_embedding=False)

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
    # if acc > best_acc and not args.debug:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not args.debug:
    #         torch.save(state, os.path.join(store_dir, 'ckpt.pth'))
    #     best_acc = acc
    return 100. * correct / total

def affnist_test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    true_correct = 0
    total = 0
    num_targets_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(affnist_test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            v = net.forward(inputs, return_embedding=False)

            loss = affnist_loss_func(v, targets)

            test_loss += loss.item()

            predicted = affnist_predicted_lambda(v)

            num_total = affnist_total_lambda(targets)
            num_targets = affnist_num_targets_lambda(targets)
            total += num_total
            num_targets_total += num_targets

            correct += affnist_correct_lambda(predicted, targets)
            true_positive_count = (predicted.eq(targets) & targets.eq(1)).sum().item()
            true_correct += true_positive_count

            if args.interactive:
                s = 'affNIST Loss: %.3f | Acc: %.3f%% (%d/%d) | Tp: %.3f (%d / %d)'
                progress_bar(batch_idx, len(test_loader), s % (
                    test_loss / (batch_idx + 1),
                    100. * correct / total, correct, total,
                    100. * true_correct / num_targets_total,
                    true_correct,
                    num_targets_total))
            else:
                print("affNIST Val Epoch:{} {}/{} Loss: {:.3f} | Acc: {:.3f}% ({}/{}) | Tp: {:.3f} ({}/{})".format(
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
    'affnist_test_acc': [],
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
    results['affnist_test_acc'].append(affnist_test(epoch))
    if not args.debug:
        pickle.dump(results, open(store_file, 'wb'))
# -
