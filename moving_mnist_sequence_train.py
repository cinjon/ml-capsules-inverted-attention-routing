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
from src.moving_mnist.moving_mnist import MovingMNist
from src import capsule_model
# from utils import progress_bar


def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
parser.add_argument('--dataset',
                    default='MovingMNist',
                    type=str,
                    help='dataset. so far only MovingMNist.')
parser.add_argument('--backbone',
                    default='resnet',
                    type=str,
                    help='type of backbone. simple or resnet')
parser.add_argument(
    '--data_root',
    default=('/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/'
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
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--epoch', default=350, type=int, help='number of epoch')
parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    help='number of batch size')
# -

args = parser.parse_args()
assert args.num_routing > 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_negative_dis = 0  # best negative distance
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Data
print('==> Preparing data..')
config = getattr(configs, args.config).config

if args.dataset == 'MovingMNist':
    image_dim_size = 64

    train_set = MovingMNist(args.data_root, train=True, sequence=True)
    test_set = MovingMNist(args.data_root, train=False, sequence=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers)
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
                              sequential_routing=args.sequential_routing,
                              return_embedding=True,
                              flatten=False)

# +
optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=0.9,
                      weight_decay=args.weight_decay)

lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[150, 250],
                                                gamma=0.1)

criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# print(net)
total_params = count_parameters(net)
# print(total_params)

results_dir = './contrastive_results'
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
    best_negative_dis = checkpoint['best_negative_dis']
    start_epoch = checkpoint['epoch']


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.
    total_positive_dis = 0.
    total_negative_dis = 0.
    t = time.time()
    for batch_idx, (inputs, positive) in enumerate(train_loader):
        inputs = inputs.to(device)
        positive = positive.to(device)

        optimizer.zero_grad()

        inputs_v = net(inputs)
        positive_v = net(positive)
        # TODO: if batch size is odd num then it won't work
        negative_v = positive_v.flip(0)

        loss = criterion(inputs_v, positive_v, negative_v)

        loss.backward()
        optimizer.step()

        t1 = time.time()
        train_loss += loss.item()
        print("t1:", time.time() - t1)

        # Compute distance
        with torch.no_grad():
            positive_dis = float(torch.dist(inputs_v, positive_v, 2).item())
            total_positive_dis += positive_dis
            negative_dis = float(torch.dist(inputs_v, negative_v, 2).item())
            total_negative_dis += negative_dis

        log_text = ('Train Epoch {} {}/{} {:.3f}s | Loss: {:.5f} |'
                    'Positive distance: {:.5f} | Negative distance: {:.5f}')
        print(
            log_text.format(epoch + 1, batch_idx + 1, len(train_loader),
                            time.time() - t, train_loss / (batch_idx + 1),
                            positive_dis, negative_dis))
        t = time.time()

    return train_loss


def test(epoch):
    global best_negative_dis
    net.eval()
    test_loss = 0
    total_positive_dis = 0.
    total_negative_dis = 0.
    t = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, positive) in enumerate(test_loader):
            inputs = inputs.to(device)
            positive = positive.to(device)

            inputs_v = net(inputs)
            positive_v = net(positive)
            negative_v = positive_v.flip(0)

            loss = criterion(inputs_v, positive_v, negative_v)

            test_loss += loss.item()

            # Compute distance
            positive_dis = float(torch.dist(inputs_v, positive_v, 2).item())
            total_positive_dis += positive_dis
            negative_dis = float(torch.dist(inputs_v, negative_v, 2).item())
            total_negative_dis += negative_dis

            log_text = ('val Epoch {} {}/{} {:.3f}s | Loss: {:.5f} |'
                        'Positive distance: {:.5f} | Negative distance: {:.5f}')
            print(
                log_text.format(epoch + 1, batch_idx + 1, len(test_loader),
                                time.time() - t, test_loss / (batch_idx + 1),
                                positive_dis, negative_dis))
            t = time.time()

    # Save checkpoint.
    if total_negative_dis > best_negative_dis and not args.debug:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'total_positive_dis': total_positive_dis,
            'total_negative_dis': total_negative_dis,
            'epoch': epoch,
        }
        if not args.debug:
            torch.save(state, os.path.join(store_dir, 'ckpt.pth'))
        best_negative_dis = total_negative_dis

    state = {'net': net.state_dict(), 'loss': test_loss, 'epoch': epoch}
    if not args.debug:
        torch.save(state, os.path.join(store_dir, 'ckpt.pth'))

    return test_loss


results = {
    'total_params': total_params,
    'args': args,
    'params': config['params'],
    'train_loss': [],
    'test_loss': [],
}

total_epochs = args.epoch

if not args.debug:
    store_file = 'dataset_%s_num_routing_%s_backbone_%s.dct' % (str(
        args.dataset), str(args.num_routing), args.backbone)
    store_file = os.path.join(store_dir, store_file)

for epoch in range(start_epoch, start_epoch + total_epochs):
    results['train_loss'].append(train(epoch))

    lr_decay.step()
    results['test_loss'].append(test(epoch))
    if not args.debug:
        pickle.dump(results, open(store_file, 'wb'))
