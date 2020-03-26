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
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_loaders(args):
    if args.dataset == 'MovingMNist':        
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
    return train_loader, test_loader


# Training
def train(epoch, net, optimizer, criterion, loader, args, device):
    print('\n***\nStarted training on epoch %d.\n***\n' % epoch)

    net.train()
    total_positive_distance = 0.
    total_negative_distance = 0.

    averages = {
        'loss': Averager()
    }
    if criterion == 'nce':
        averages['pos_sim'] = Averager()
        averages['neg_sim'] = Averager()
    elif criterion == 'xent':
        averages['true_pos'] = Averager()
        averages['num_targets'] = Averager()
        true_positive_total = 0
        num_targets_total = 0

    t = time.time()
    optimizer.zero_grad()
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)

        if criterion == 'triplet':
            # NOTE: Zeping.
            loss, stats = net.get_triplet_loss(images)
            averages['loss'].add(loss.item())
            positive_distance = stats['positive_distance']
            negative_distance = stats['negative_distance']
            total_positive_distance += positive_distance
            total_negative_distance += negative_distance
            extra_s = 'Pos distance: {:.5f} | Neg distance: {:.5f}'.format(
                positive_distance, negative_distance
            )
        elif criterion == 'xent':
            labels = labels.to(device)
            loss, stats = net.get_xent_loss(images, labels)
            averages['loss'].add(loss.item())
            true_positive_total += stats['true_pos']
            num_targets_total += stats['num_targets']
            extra_s = 'True Pos Rate: {:.5f} ({} / {}).'.format(
                100. * true_positive_total / num_targets_total,
                true_positive_total, num_targets_total
            )            
        elif criterion == 'nce':
            loss, stats = net.get_nce_loss(images, args)
            averages['loss'].add(loss.item())
            for key, value in stats.items():
                averages[key].add(value)
            extra_s = 'Pos sim: {:.5f} | Neg sim: {:.5f}.'.format(
                averages['pos_sim'].item(), averages['neg_sim'].item()
            )
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 100 == 0:
            log_text = ('Train Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
            print(log_text.format(epoch, batch_idx, len(loader),
                                  time.time() - t, averages['loss'].item())
            )
            t = time.time()
            
    train_loss = averages['loss'].item()
    return train_loss


def test(epoch, net, criterion, loader, args, best_negative_distance, device):
    print('\n***\nStarted test on epoch %d.\n***\n' % epoch)

    net.eval()
    total_positive_distance = 0.
    total_negative_distance = 0.

    averages = {
        'loss': Averager()
    }
    if criterion == 'nce':
        averages['pos_sim'] = Averager()
        averages['neg_sim'] = Averager()
    elif criterion == 'xent':
        averages['true_pos'] = Averager()
        averages['num_targets'] = Averager()
        true_positive_total = 0
        num_targets_total = 0

    t = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)

            if criterion == 'triplet':
                # NOTE: Zeping.
                loss, stats = net.get_triplet_loss(images)
                averages['loss'].add(loss.item())
                positive_distance = stats['positive_distance']
                negative_distance = stats['negative_distance']
                total_positive_distance += positive_distance
                total_negative_distance += negative_distance
                extra_s = 'Pos distance: {:.5f} | Neg distance: {:.5f}'.format(
                    positive_distance, negative_distance
                )
            elif criterion == 'xent':
                labels = labels.to(device)
                loss, stats = net.get_xent_loss(images, labels)
                averages['loss'].add(loss.item())
                true_positive_total += stats['true_pos']
                num_targets_total += stats['num_targets']
                extra_s = 'True Pos Rate: {:.5f} ({} / {}).'.format(
                    100. * true_positive_total / num_targets_total,
                    true_positive_total, num_targets_total
                )            
            elif criterion == 'nce':
                loss, stats = net.get_nce_loss(images, args)
                averages['loss'].add(loss.item())
                for key, value in stats.items():
                    averages[key].add(value)
                extra_s = 'Pos sim: {:.5f} | Neg sim: {:.5f}.'.format(
                    averages['pos_sim'].item(), averages['neg_sim'].item()
                )

            if batch_idx % 100 == 0:
                log_text = ('Val Epoch {} {}/{} {:.3f}s | Loss: {:.5f} | ' + extra_s)
                print(log_text.format(epoch, batch_idx, len(loader),
                                      time.time() - t, averages['loss'].item())
                )
                t = time.time()

    test_loss = averages['loss'].item()

    # Save checkpoint.
    state = {
        'net': net.state_dict(),
        'total_positive_distance': total_positive_distance,
        'total_negative_distance': total_negative_distance,
        'epoch': epoch,
        'loss': test_loss
    }
    if total_negative_distance > best_negative_distance:
        print('Total Neg Distance better than before.')
        if not args.debug:
            torch.save(state, os.path.join(store_dir, 'ckpt.pth'))
        best_negative_distance = total_negative_distance
    elif not args.debug and epoch % 20 == 0:
        torch.save(state, os.path.join(store_dir, 'ckpt.%d.pth' % epoch))

    return test_loss, total_negative_distance


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

    train_loader, test_loader = get_loaders(args)

    print('==> Building model..')
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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[150, 250],
                                                         gamma=0.1)
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
        'train_loss': [],
        'test_loss': [],
    }

    total_epochs = args.epoch
    if not args.debug:
        store_file = 'dataset_%s_num_routing_%s_backbone_%s.dct' % (
            str(args.dataset), str(args.num_routing), args.backbone
        )
        store_file = os.path.join(store_dir, store_file)

    # test_loss, total_negative_distance = test(0, net, args.criterion, test_loader, args, best_negative_distance, device)
    for epoch in range(start_epoch, start_epoch + total_epochs):
        train_loss = train(epoch, net, optimizer, args.criterion, train_loader, args, device)
        results['train_loss'].append(train_loss)

        if scheduler:
            scheduler.step()

        test_loss, total_negative_distance = test(epoch, net, args.criterion, test_loader, args, best_negative_distance, device)
        best_negative_distance = max(best_negative_distance, total_negative_distance)
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
                        help='triplet, nce, or xent.')
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

    args = parser.parse_args()
    assert args.num_routing > 0
    main(args)
