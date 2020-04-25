import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18

class ReorderResNet(nn.Module):
    def __init__(self, resnet_type='resnet50', pretrained=True):
        super(ReorderResNet, self).__init__()
        self.resnet_type = resnet_type
        self.pretrained = pretrained

        if self.resnet_type == 'resnet50':
            resnet = resnet50(self.pretrained)
            self.resnet_out_channels = 2048
        else:
            resnet = resnet18(self.pretrained)
            self.resnet_out_channels = 512
        self.res = nn.Sequential(*list(resnet.children())[:-1])
        self.predictor = nn.Linear(self.resnet_out_channels * 3, 1)

    def get_feature(self, x):
        x = self.res(x)
        x = x.view(-1, self.resnet_out_channels)
        return x

    def get_ordering(self, x):
        x = self.predictor(x)
        return x

    def get_reorder_loss(self, images, device, args, labels):
        images_shape = images.shape

        batch_size, num_images = images.shape[:2]

        # Change view so that we can put everything through the model at once.
        images = images.view(batch_size * num_images, *images.shape[2:])
        pose = self.get_feature(images)
        pose = pose.view(batch_size, num_images, *pose.shape[1:])

        # Get the ordering.
        flattened = pose.view(batch_size, -1)
        ordering = self.get_ordering(flattened).squeeze(-1)

        loss_ordering = F.binary_cross_entropy_with_logits(ordering, labels)
        predictions = torch.sigmoid(ordering) > 0.5
        # DEBUG
        # print('{:.3f}'.format(
        #     torch.sum(predictions.detach().cpu()).item() / len(predictions)))
        accuracy = (predictions == labels).float().mean().item()

        total_loss = sum([
            args.lambda_ordering * loss_ordering,
            # args.lambda_object_sim * loss_objects
        ])

        stats = {
            'accuracy': accuracy,
            # 'objects_sim_loss': loss_objects.item(),
            # 'presence_sparsity_loss': loss_sparsity.item(),
            'ordering_loss': loss_ordering.item()
        }
        return total_loss, stats
