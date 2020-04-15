import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ReorderResNet(nn.Module):
    def __init__(self):
        super(ReorderResNet, self).__init__()
        self.res = nn.Sequential(*list(resnet50().children())[:-1])
        self.predictor = nn.Linear(2048 * 3, 1)

    def get_feature(self, x):
        x = self.res(x)
        x = x.view(-1, 2048)
        return x

    def get_ordering(self, x):
        x = self.predictor(x)
        return x

    def get_new_reorder_loss(self, images, labels, args):
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

    def get_reorder_loss(self, images, device, args, labels=None):
        # images come in as [bs, num_imgs, ch, w, h]. we want to pick from
        # this three frames to use as either positive or negative.

        # select frames (a, b, c, d, e)
        range_size = int(images.shape[1] / 5)

        sample = [random.choice(range(i*range_size, (i+1)*range_size))
                  for i in range(5)]

        # Maybe flip the list's order.
        if random.random() > 0.5:
            sample = list(reversed(sample))

        use_positive = random.random() > 0.5
        if use_positive:
            # frames (b, c, d) or (d, c, b)
            selection = sample[1:4]
        elif random.random() > 0.5:
            # frames (b, a, d), (d, a, b), (b, e, d), or (d, e, b)
            selection = [sample[1], sample[0], sample[3]]
        else:
            selection = [sample[1], sample[4], sample[3]]

        images = images[:, selection]

        images_shape = images.shape

        batch_size, num_images = images.shape[:2]
        images = images.to(device)
        labels = torch.tensor([use_positive]*batch_size).type(torch.FloatTensor).to(images.device)

        # Change view so that we can put everything through the model at once.
        images = images.view(batch_size * num_images, *images.shape[2:])
        pose = self.get_feature(images)
        pose = pose.view(batch_size, num_images, *pose.shape[1:])

        # Get the ordering.
        # flattened = object_pose_presence.view(batch_size, -1)
        flattened = pose.view(batch_size, -1)
        ordering = self.get_ordering(flattened).squeeze(-1)

        loss_ordering = F.binary_cross_entropy_with_logits(ordering, labels)
        predictions = torch.sigmoid(ordering) > 0.5
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
