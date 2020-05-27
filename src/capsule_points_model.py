#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import os
import random

import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch

from src import layers_1d


# Capsule model
class CapsulePointsModel(nn.Module):

    def __init__(self, params, args):
        super(CapsulePointsModel, self).__init__()
        #### Parameters

        self.num_routing = args.num_routing  # >3 may cause slow converging
        self.presence_type = args.presence_type

        #### Backbone is a ResNet that embeds each point.
        backbone = params['backbone']
        self.pre_caps = layers_1d.ResnetBackbone(backbone['input_dim'],
                                                 backbone['output_dim'],
                                                 backbone['stride'])

        ## Primary Capsule Layer. In the same spirit as the image version, we
        # use a Conv here, but it has to be 1d. Maybe this should be a set
        # transformer?
        primary = params['primary_capsules']
        self.pc_num_caps = primary['num_caps']
        self.pc_caps_dim = primary['caps_dim']
        self.pc_output_dim = primary['out_img_size']
        self.pc_layer = nn.Conv1d(
            in_channels=primary['input_dim'],
            out_channels=primary['num_caps'] * primary['caps_dim'],
            kernel_size=primary['kernel_size'],
            stride=primary['stride'],
            padding=primary['padding'],
            bias=False)
        self.layer_norm = nn.LayerNorm(primary['caps_dim'])

        ## Main Capsule Layers
        self.capsule_layers = nn.ModuleList([])
        capsules = params['capsules']
        for i in range(len(capsules)):
            if capsules[i]['type'] == 'CONV':
                if i == 0:
                    in_n_caps = primary['num_caps']
                    in_d_caps = primary['caps_dim']
                else:
                    in_n_caps = capsules[i-1]['num_caps']
                    in_d_caps = capsules[i-1]['caps_dim']

                self.capsule_layers.append(
                    layers_1d.CapsuleConv(
                        in_n_capsules=in_n_caps,
                        in_d_capsules=in_d_caps,
                        out_n_capsules=capsules[i]['num_caps'],
                        out_d_capsules=capsules[i]['caps_dim'],
                        kernel_size=capsules[i]['kernel_size'],
                        stride=capsules[i]['stride'],
                    )
                )
            elif capsules[i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = primary['num_caps'] * primary['out_img_size']
                    in_d_caps = primary['caps_dim']
                elif capsules[i - 1]['type'] == 'FC':
                    in_n_caps = capsules[i - 1]['num_caps']
                    in_d_caps = capsules[i - 1]['caps_dim']
                elif capsules[i - 1]['type'] == 'CONV':
                    in_n_caps = capsules[i-1]['num_caps'] * capsules[i-1]['out_img_size']
                    in_d_caps = capsules[i - 1]['caps_dim']
                self.capsule_layers.append(
                    layers_1d.CapsuleFC(
                        in_n_capsules=in_n_caps,
                        in_d_capsules=in_d_caps,
                        out_n_capsules=capsules[i]['num_caps'],
                        out_d_capsules=capsules[i]['caps_dim']
                    )
                )

        ## Class Capsule Layer
        if not len(capsules) == 0:
            if capsules[-1]['type'] == 'FC':
                in_n_caps = capsules[-1]['num_caps']
                in_d_caps = capsules[-1]['caps_dim']
            elif capsules[-1]['type'] == 'CONV':
                in_n_caps = capsules[-1]['num_caps'] * capsules[-1]['out_img_size']
                in_d_caps = capsules[-1]['caps_dim']
        else:
            in_n_caps = primary['num_caps'] * primary['out_img_size']
            in_d_caps = primary['caps_dim']
        self.capsule_layers.append(
            layers_1d.CapsuleFC(
                in_n_capsules=in_n_caps,
                in_d_capsules=in_d_caps,
                out_n_capsules=params['class_capsules']['num_caps'],
                out_d_capsules=params['class_capsules']['caps_dim'],
            )
        )

    def get_presence(self, final_pose):
        if not self.presence_type:
            return None
        elif self.presence_type == 'l2norm':
            return final_pose.norm(dim=2)

    def forward(self, x):
        # x: torch.Size([24, 3, 2048])
        c = self.pre_caps(x)    
        # precaps torch.Size([bs=24, backbone.output_dim=128, stride of 2 --> 1024])

        ## Primary Capsule Layer (a single CNN)
        # u:  torch.Size([24, 512, 1024])
        u = self.pc_layer(c)

        u = u.permute(0, 2, 1)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim)
        u = u.permute(0, 2, 1, 3)  # 100, 32, 14, 14, 16
        init_capsule_value = self.layer_norm(u)  #capsule_utils.squash(u)

        ## Main Capsule Layers
        # concurrent routing
        # first iteration
        # perform initilialization for the capsule values as single forward passing
        capsule_values, _val = [init_capsule_value], init_capsule_value
        for i in range(len(self.capsule_layers)):
            # TODO: Oh I see, so this problem happens only on CapsuleFC.
            # init capsule 0: [24, 32, 511, 16]
            # init capsule 1: [24, 32, 255, 16]
            # Note that precaps is [24,128,1024] and u is [24,512,512] and x is [24,3,2048]
            _val = self.capsule_layers[i].forward(_val, 0)
            capsule_values.append(_val)        

        # second to t iterations
        # perform the routing between capsule layers
        for n in range(self.num_routing - 1):
            _capsule_values = [init_capsule_value]
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(
                    capsule_values[i], n, capsule_values[i + 1])
                _capsule_values.append(_val)
            capsule_values = _capsule_values
                
        ## After Capsule
        pose = capsule_values[-1]
        presence = self.get_presence(pose)
        return pose, presence


class BackboneModel(nn.Module):

    def __init__(self, params, args):
        super(BackboneModel, self).__init__()

        self.num_routing = args.num_routing  # >3 may cause slow converging
        self.presence_type = args.presence_type

        #### Backbone is a ResNet that embeds each point.
        backbone = params['backbone']
        self.pre_caps = layers_1d.ResnetBackbone(backbone['input_dim'],
                                                 backbone['output_dim'],
                                                 backbone['stride'])

        input_dim = backbone['output_dim']
        input_dim *= int(backbone['inp_img_size']/2)
        output_dim = 5 if args.dataset == 'shapenet5' else 55
        self.fc_head = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        c = self.pre_caps(x)
        c = c.view(c.shape[0], -1)
        out = self.fc_head(c)
        return out


def get_xent_loss(model, points, labels):
    output = model(points, return_embedding=False)
    loss = F.cross_entropy(output, labels)
    predictions = torch.argmax(output, dim=1)
    accuracy = (predictions == labels).float().mean().item()
    stats = {
        'accuracy': accuracy,
    }
    return loss, stats


def get_backbone_test_loss(model, images, labels, args):
    # Images are expected to come as singular, not as [bs, num_images].
    output = model(images)
    loss = F.cross_entropy(output, labels)
    predictions = torch.argmax(output, dim=1)
    accuracy = (predictions == labels).float().mean().item()
    stats = {
        'accuracy': accuracy,
    }
    return loss, stats


def do_simclr_nce(temperature, probs=None, anchor=None, other=None,
                  suffix='presence', selection_strategy='anchor_other12',
                  do_norm=True):
    """NOTE: This normalization that happens here should not be applied
    across capsules, only across probs or an individiaul capsule.
    """
    if selection_strategy == 'default':
        selection = [0, 1]
    elif selection_strategy == 'randomize_selection':
        selection = np.random.randint(0, probs.shape[1], [2])
    elif selection_strategy == 'anchor0_other12':
        selection = np.random.randint(1, probs.shape[1], [1])
        selection = [0, selection[0]]
    else:
        raise

    if anchor is None and other is None:
        anchor = probs[:, selection[0]]
        other = probs[:, selection[1]]
    elif anchor is None or other is None:
        raise

    batch_size = anchor.shape[0]

    if do_norm:
        anchor = F.normalize(anchor, dim=1)
        other = F.normalize(other, dim=1)
    representations = torch.cat([other, anchor], dim=0)
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    # filter out the scores from the positive samples
    l_pos = torch.diag(similarity_matrix, batch_size)
    r_pos = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

    # the masking
    diag = np.eye(2 * batch_size)
    l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
    l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
    mask = torch.from_numpy((diag + l1 + l2))
    mask = (1 - mask).type(torch.bool)
    mask = mask.to(anchor.device)

    negatives = similarity_matrix[mask].view(2*batch_size, -1)
    neg_similarity = negatives.mean()
    pos_similarity = positives.mean()

    logits = torch.cat((positives, negatives), dim=1)
    logits /= temperature

    labels = torch.zeros(2 * batch_size).to(anchor.device).long()
    loss = F.cross_entropy(logits, labels, reduction='mean')
    # loss = loss / (2 * batch_size)

    stats = {
        'nce_%s' % suffix: loss.item(),
        'pos_sim_%s' % suffix: pos_similarity.item(),
        'neg_sim_%s' % suffix: neg_similarity.item(),
    }
    return loss, stats


def get_nceprobs_selective_loss(model, points, device, epoch, args,
                                num_class_capsules=10):
    """Get the loss with the nce probs approach.

    We take the nce over the probs. Then we detach those probs and use them in
    *some* way in order to inform other losses.

    An example would be ncelinear_max, which takes the maximally activated
    capsule and uses that in *another* nce that tries to make same clips linear
    and differing clips not linear.

    Another would be ncelinear_thresh, which takes all capsules that are
    activated above a certain threshold (say .9) in their probs and then does
    the sum of each of those NCEs.
    """
    # Change view so that we can put everything through the model at once.
    batch_size, num_points = points.shape[:2]
    points = points.view(batch_size * num_points, *points.shape[2:])
    pose, presence = model(points)
    pose = pose.view(batch_size, num_points, *pose.shape[1:])
    presence_point = presence.view(batch_size, num_points, *presence.shape[1:])

    stats = {}
    loss = 0.

    # Get the loss for the nce over probs.
    nce, stats_ = do_simclr_nce(
        args.nce_presence_temperature, presence_point, do_norm=True,
        selection_strategy=args.simclr_selection_strategy)
    loss += nce * args.nce_presence_lambda
    stats.update(stats_)

    presence_first = presence_point[:, 0]
    max_values, _ = torch.max(presence_first, 1)
    min_values, _ = torch.min(presence_first, 1)
    mean_per_capsule = [value.item() for value in presence.mean(0)]
    std_per_capsule = [value.item() for value in presence.std(0)]
    l2_probs_12 = torch.norm(
        presence_point[:, 0] - presence_point[:, 1], dim=1).mean()
    
    for num, item in enumerate(mean_per_capsule):
        stats['capsule_prob_mean_%d' % num] = item
    for num, item in enumerate(std_per_capsule):
        stats['capsule_prob_std_%d' % num] = item

    stats.update({
        'mean_max_l2norm': max_values.mean().item(),
        'mean_min_l2norm': min_values.mean().item(),
        'mean_l2norm': presence_first.mean().item(),
        'l2_presence_l2norms_12': l2_probs_12.item(),
    })

    # Get the frame distance stats.
    sim_12 = torch.dist(pose[:, 0, :], pose[:, 1, :])
    stats.update({
        'frame_12_sim': sim_12.item(),
    })

    if presence_point.shape[1] == 3:
        l2_probs_13 = torch.norm(
            presence_point[:, 0] - presence_point[:, 2], dim=1).mean()
        l2_probs_23 = torch.norm(
            presence_point[:, 1] - presence_point[:, 2], dim=1).mean()
        sim_23 = torch.dist(pose[:, 1, :], pose[:, 2, :])
        sim_13 = torch.dist(pose[:, 0, :], pose[:, 2, :])
        segment_12 = pose[:, 1, :] - pose[:, 0, :]
        segment_13 = pose[:, 2, :] - pose[:, 0, :]
        segment_23 = pose[:, 2, :] - pose[:, 1, :]
        stats.update({
            'l2_presence_l2norms_13': l2_probs_13.item(),
            'l2_presence_l2norms_23': l2_probs_23.item(),
            'frame_23_sim': sim_23.item(),
            'frame_13_sim': sim_13.item(),
            'triangle_margin': (sim_13 - sim_12 - sim_23).item(),
        })

    return loss, stats
