import argparse
from datetime import datetime
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import configs
from src.moving_mnist import MovingMNist, MovingMNist_collate
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
    default='MovingMNist',
    type=str,
    help='dataset. CIFAR10, CIFAR100, DiverseMultiMNist, MovingMNist.')
parser.add_argument('--backbone',
                    default='resnet',
                    type=str,
                    help='type of backbone. simple or resnet')
parser.add_argument('--data_root',
                    default=('/misc/kcgscratch1/ChoGroup/resnick/'
                             'spaceofmotion/zeping/capsules/'
                             'ml-capsules-inverted-attention-routing/data'),
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
                    help='number of epoch')
parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    help='number of batch size')
# -

args = parser.parse_args()
assert args.num_routing > 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
assert args.dataset in [
    'CIFAR10', 'CIFAR100', 'MovingMNist', 'DiverseMultiMNist'
]

config = getattr(configs, args.config).config

if args.dataset == 'MovingMNist':
    # train_batch_size = 128
    # test_batch_size = 128
    image_dim_size = 64

    random.seed(args.seed)
    dataset = MovingMNist(args.data_root, download=True)
    train_len = int(len(dataset) * 0.8)
    test_len = len(dataset) - train_len
    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_len, test_len])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=MovingMNist_collate)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=MovingMNist_collate)
else:
    raise

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

criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# -


def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(net)
total_params = count_parameters(net)
print(total_params)

results_dir = './results'
if not os.path.isdir(results_dir) and not args.debug:
    os.mkdir(results_dir)

today = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
store_dir = os.path.join(results_dir, today)
if not args.debug:
    os.mkdir(store_dir)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


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
    for batch_idx, (inputs, positive, negative) in enumerate(train_loader):
        inputs = inputs.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        inputs_v = net(inputs)
        positive_v = net(positive)
        negative_v = net(negative)

        loss = criterion(inputs_v, positive_v, negative_v)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    #     predicted = predicted_lambda(v)
    #
    #     num_total = total_lambda(targets)
    #     num_targets = num_targets_lambda(targets)
    #     total += num_total
    #     num_targets_total += num_targets
    #
    #     correct += correct_lambda(predicted, targets)
    #     true_positive_count = (predicted.eq(targets) & targets.eq(1)).sum().item()
    #     true_correct += true_positive_count
    #
    #     s = 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Tp: %.3f (%d / %d)'
    #
    #     progress_bar(batch_idx, len(train_loader), s % (
    #         train_loss / (batch_idx + 1),
    #         100. * correct / total, correct, total,
    #         100. * true_correct / num_targets_total,
    #         true_correct,
    #         num_targets_total
    #     ))

        print("Epoch {} {} / {} | Loss: {:.5f}".format(
            epoch, batch_idx, len(train_loader), train_loss/(batch_idx+1)))

    return train_loss # 100. * correct / total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    true_correct = 0
    total = 0
    num_targets_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, positive, negative) in enumerate(test_loader):
            inputs = inputs.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            inputs_v = net(inputs)
            positive_v = net(positive)
            negative_v = net(negative)

            loss = criterion(inputs_v, positive_v, negative_v)

            test_loss += loss.item()

            # predicted = predicted_lambda(v)
            #
            # num_total = total_lambda(targets)
            # num_targets = num_targets_lambda(targets)
            # total += num_total
            # num_targets_total += num_targets
            #
            # correct += correct_lambda(predicted, targets)
            # true_positive_count = (predicted.eq(targets) & targets.eq(1)).sum().item()
            # true_correct += true_positive_count

            # s = 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Tp: %.3f (%d / %d)'
            #
            # progress_bar(batch_idx, len(test_loader), s % (
            #     test_loss / (batch_idx + 1),
            #     100. * correct / total, correct, total,
            #     100. * true_correct / num_targets_total,
            #     true_correct,
            #     num_targets_total
            # ))

            print("Epoch {} {} / {} | Loss: {:.5f}".format(
                epoch, batch_idx, len(test_loader), test_loss/(batch_idx+1)))

    # Save checkpoint.
    # acc = 100. * correct / total
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

    state = {
        "net": net.state_dict(),
        "loss": test_loss,
        "epoch": epoch
    }
    if not args.debug:
        torch.save(state, os.path.join(store_dir, "ckpt.pth"))

    return test_loss # 100. * correct / total


# +
results = {
    'total_params': total_params,
    'args': args,
    'params': config['params'],
    'train_loss': [],
    'test_loss': [],
}

total_epochs = args.epoch

if not args.debug:
    store_file = 'dataset_%s_num_routing_%s_backbone_%s.dct' % (
        str(args.dataset), str(args.num_routing), args.backbone)
    store_file = os.path.join(store_dir, store_file)

for epoch in range(start_epoch, start_epoch + total_epochs):
    results['train_loss'].append(train(epoch))

    lr_decay.step()
    results['test_loss'].append(test(epoch))
    print("Epoch {} total test loss: {}".format(results['test_loss'][-1]))
    if not args.debug:
        pickle.dump(results, open(store_file, 'wb'))
# -
