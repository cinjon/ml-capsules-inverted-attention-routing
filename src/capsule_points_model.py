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

    def __init__(self,
                 params,
                 backbone,
                 num_routing,
    ):
        super(CapsTimeModel, self).__init__()
        #### Parameters

        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        self.num_routing = num_routing  # >3 may cause slow converging

        #### Building Networks
        ## Backbone (before capsule)
        self.pre_caps = layers_1d.resnet_backbone(
            params['backbone']['input_dim'],
            params['backbone']['output_dim'],
            params['backbone']['stride'])

        self.num_class_capsules = params['class_capsules']['num_caps']
        self.is_discriminating_model = is_discriminating_model

        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                  out_channels=params['primary_capsules']['num_caps'] *\
                                  params['primary_capsules']['caps_dim'],
                                  kernel_size=params['primary_capsules']['kernel_size'],
                                  stride=params['primary_capsules']['stride'],
                                  padding=params['primary_capsules']['padding'],
                                  bias=False)

        self.nonlinear_act = nn.LayerNorm(
            params['primary_capsules']['caps_dim'])

        ## Main Capsule Layers
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.CapsuleCONV(
                        in_n_capsules=in_n_caps,
                        in_d_capsules=in_d_caps,
                        out_n_capsules=params['capsules'][i]['num_caps'],
                        out_d_capsules=params['capsules'][i]['caps_dim'],
                        kernel_size=params['capsules'][i]['kernel_size'],
                        stride=params['capsules'][i]['stride'],
                        matrix_pose=params['capsules'][i]['matrix_pose'],
                        dp=dp,
                        coordinate_add=False))
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i - 1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i - 1]['num_caps']
                    in_d_caps = params['capsules'][i - 1]['caps_dim']
                elif params['capsules'][i - 1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']
                    in_d_caps = params['capsules'][i - 1]['caps_dim']
                self.capsule_layers.append(
                    layers.CapsuleFC(
                        in_n_capsules=in_n_caps,
                        in_d_capsules=in_d_caps,
                        out_n_capsules=params['capsules'][i]['num_caps'],
                        out_d_capsules=params['capsules'][i]['caps_dim'],
                        matrix_pose=params['capsules'][i]['matrix_pose'],
                        dp=dp))

        ## Class Capsule Layer
        if not len(params['capsules']) == 0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            layers.CapsuleFCPresenceObject(
                in_n_capsules=in_n_caps,
                in_d_capsules=in_d_caps,
                out_n_capsules=params['class_capsules']['num_caps'],
                out_d_capsules=params['class_capsules']['caps_dim'],
                matrix_pose=params['class_capsules']['matrix_pose'],
                dp=dp,
                object_dim=params['class_capsules']['object_dim'])
        )

    def get_presence(self, pre_caps, final_pose):
        # NOTE: This is assuming we are using l2norm as the presence probs
        logits = final_pose.norm(dim=2)
        return logits

    def forward(self, x):
        c = self.pre_caps(x)

        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c)
        # dmm u.shape: [64, 1024, 8, 8]
        u = u.permute(0, 2, 3, 1)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim,
                   self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4)  # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)  #capsule_utils.squash(u)
        
        ## Main Capsule Layers
        # concurrent routing
        # first iteration
        # perform initilialization for the capsule values as single forward passing
        capsule_values, _val = [init_capsule_value], init_capsule_value
        for i in range(len(self.capsule_layers)):
            _val = self.capsule_layers[i].forward(_val, 0)
            # dmm:
            # capsule 0 _val.shape: [64, 16, 6, 6, 64]
            # capsule 1 _val.shape: [64, 10, 64]
            # capsule 2 _val.shape: [64, 10, 64]
            # get the capsule value for next layer
            capsule_values.append(_val)
            
        # second to t iterations
        # perform the routing between capsule layers
        for n in range(self.num_routing - 1):
            _capsule_values = [init_capsule_value]
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(
                    capsule_values[i], n, capsule_values[i + 1])
                # dmm:
                # capsule 0 _val.shape: [64, 16, 6, 6, 64]
                # capsule 1 _val.shape: [64, 10, 64]
                # capsule 2 _val.shape: [64, 10, 64]
                _capsule_values.append(_val)
            capsule_values = _capsule_values
                
        ## After Capsule
        pose = capsule_values[-1]
        return pose


def get_xent_loss(model, points, labels):
    output = model(points, return_embedding=False)
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
    pose, presence_probs = model(points)
    pose = pose.view(batch_size, num_points, *pose.shape[1:])
    presence_probs_point = presence_probs.view(
        batch_size, num_points, -1)

    stats = {}
    loss = 0.

    # Get the loss for the nce over probs.
    nce, stats_ = do_simclr_nce(
        args.nce_presence_temperature, presence_probs_point,
        selection_strategy=args.simclr_selection_strategy,
        do_norm=args.simclr_do_norm)        
    loss += nce * args.nce_presence_lambda
    stats.update(stats_)

    presence_probs_first = presence_probs_point[:, 0]
    max_values, _ = torch.max(presence_probs_first, 1)
    min_values, _ = torch.min(presence_probs_first, 1)
    mean_per_capsule = [value.item() for value in presence_probs.mean(0)]
    std_per_capsule = [value.item() for value in presence_probs.std(0)]
    l2_probs_12 = torch.norm(
        presence_probs_point[:, 0] - presence_probs_point[:, 1], dim=1).mean()
    l2_probs_13 = torch.norm(
        presence_probs_point[:, 0] - presence_probs_point[:, 2], dim=1).mean()
    l2_probs_23 = torch.norm(
        presence_probs_point[:, 1] - presence_probs_point[:, 2], dim=1).mean()
    
    for num, item in enumerate(mean_per_capsule):
        stats['capsule_prob_mean_%d' % num] = item
    for num, item in enumerate(std_per_capsule):
        stats['capsule_prob_std_%d' % num] = item

    stats.update({
        'mean_max_l2norm': max_values.mean().item(),
        'mean_min_l2norm': min_values.mean().item(),
        'mean_l2norm': presence_probs_first.mean().item(),
        'l2_presence_l2norms_12': l2_probs_12.item(),
        'l2_presence_l2norms_13': l2_probs_13.item(),
        'l2_presence_l2norms_23': l2_probs_23.item(),
    })

    # Get the frame distance stats.
    sim_12 = torch.dist(pose[:, 0, :], pose[:, 1, :])
    sim_23 = torch.dist(pose[:, 1, :], pose[:, 2, :])
    sim_13 = torch.dist(pose[:, 0, :], pose[:, 2, :])
    segment_12 = pose[:, 1, :] - pose[:, 0, :]
    segment_13 = pose[:, 2, :] - pose[:, 0, :]
    segment_23 = pose[:, 2, :] - pose[:, 1, :]
    stats.update({
        'frame_12_sim': sim_12.item(),
        'frame_23_sim': sim_23.item(),
        'frame_13_sim': sim_13.item(),
        'triangle_margin': (sim_13 - sim_12 - sim_23).item(),
    })

    return loss, stats
