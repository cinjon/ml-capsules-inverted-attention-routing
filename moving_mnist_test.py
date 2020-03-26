import os
import pickle
import argparse
import numpy as np

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

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_root',
    default='/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/',
    type=str,
    help='root of where to put the data.')
parser.add_argument(
    '--resume_dir',
    default=
    'cd /misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/result/moving_mnist_1',
    type=str,
    help='dir where we resume from checkpoint')
parser.add_argument('--subset', default='train', type=str, help='train or val')
parser.add_argument('--num_routing',
                    default=3,
                    type=int,
                    help='number of routing. Recommended: 0,1,2,3.')
parser.add_argument('--sequential_routing',
                    action='store_true',
                    help='not using concurrent_routing')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = getattr(configs, "resnet_backbone_movingmnist").config


def main():
    # Dataset
    print("Loading data")
    image_dim_size = 64
    train_batch_size = 16
    test_batch_size = 16

    train = True if args.subset == "train" else False
    dataset = MovingMNist(root=args.data_root, train=train)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=train_batch_size,
                                             shuffle=True,
                                             num_workers=12)

    # Lambda
    predicted_lambda = lambda v: (torch.sigmoid(v) > 0.5).type(v.dtype)
    # correct_lambda = lambda predicted, targets:\
    #     predicted.eq(targets).all(1).sum().item()
    # total_lambda = lambda targets: targets.size(0)
    # num_targets_lambda = lambda targets: targets.eq(1).sum().item()

    # Model parameters
    # print(config)
    net = capsule_model.CapsModel(image_dim_size,
                                  config['params'],
                                  "resnet",
                                  0,
                                  args.num_routing,
                                  sequential_routing=args.sequential_routing)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']

    # Testing
    net.eval()
    one_label_correct = 0
    one_label_total = 0
    two_label_correct = 0
    two_label_total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        v = net(inputs)
        # loss = loss_func(v, targets)
        # test_loss += loss.item()
        predicted = predicted_lambda(v)

        # Get correct labels
        correct_arr = predicted.eq(targets).all(1).cpu()
        one_label_arr = (torch.sum(targets, dim=1) == 1).cpu()
        two_label_arr = (torch.sum(targets, dim=1) == 2).cpu()

        one_label_correct += int((correct_arr & one_label_arr).sum())
        one_label_total += int(one_label_arr.sum())
        two_label_correct += int((correct_arr & two_label_arr).sum())
        two_label_total += int(two_label_arr.sum())

        print("{}/{} one label: {:.3f}% ({}/{})\t two label: {:.3f}% ({}/{})".
              format(batch_idx + 1, len(dataloader),
                     100. * one_label_correct / one_label_total,
                     one_label_correct, one_label_total,
                     100. * two_label_correct / two_label_total,
                     two_label_correct, two_label_total))


if __name__ == "__main__":
    main()
