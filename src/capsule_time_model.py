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

from src import layers


# Capsule model
class CapsTimeModel(nn.Module):

    def __init__(self,
                 params,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True,
                 mnist_classifier_head=False,
                 mnist_classifier_strategy='pose',
                 num_frames=4,
                 is_discriminating_model=False, # Use True if original model.
                 use_presence_probs=False,
                 presence_temperature=1.0,
                 presence_loss_type='sigmoid_l1',
                 do_capsule_computation=True,
                 do_discriminative_probs=False,
                 do_selective_reorder=False,
    ):
        super(CapsTimeModel, self).__init__()
        #### Parameters
        self.sequential_routing = sequential_routing

        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing  # >3 may cause slow converging

        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(
                params['backbone']['input_dim'],
                params['backbone']['output_dim'],
                params['backbone']['kernel_size'],
                params['backbone']['stride'],
                params['backbone']['padding'])
        elif backbone == 'resnet':
            self.pre_caps = layers.resnet_backbone(
                params['backbone']['input_dim'],
                params['backbone']['output_dim'],
                params['backbone']['stride'])

        self.num_class_capsules = params['class_capsules']['num_caps']
        self.is_discriminating_model = is_discriminating_model
        self.use_presence_probs = use_presence_probs
        self.presence_temperature = presence_temperature
        self.presence_loss_type = presence_loss_type
        if use_presence_probs:
            if 'squash' not in presence_loss_type and 'l2norm' not in presence_loss_type:
                input_dim = params['backbone']['output_dim']
                input_dim *= int(params['backbone']['inp_img_size']/2)**2
                output_dim = params['class_capsules']['num_caps']
                self.presence_prob_head = nn.Linear(input_dim, output_dim)

        self.do_discriminative_probs = do_discriminative_probs
        if self.do_discriminative_probs:
            input_dim = params['class_capsules']['num_caps']
            output_dim = 10
            self.final_fc_probs = nn.Linear(input_dim, output_dim)

        self.do_capsule_computation = do_capsule_computation
        if not do_capsule_computation:
            self.dummy_pose = torch.randn(
                params['class_capsules']['num_caps'],
                params['class_capsules']['caps_dim'],
                dtype=torch.float
            )
            return

        self.do_selective_reorder = do_selective_reorder

        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                  out_channels=params['primary_capsules']['num_caps'] *\
                                  params['primary_capsules']['caps_dim'],
                                  kernel_size=params['primary_capsules']['kernel_size'],
                                  stride=params['primary_capsules']['stride'],
                                  padding=params['primary_capsules']['padding'],
                                  bias=False)

        #self.pc_layer = nn.Sequential()

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

        if is_discriminating_model:
            ## After Capsule
            # fixed classifier for all class capsules
            self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
            # different classifier for different capsules
            #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))
        else:
            if self.do_selective_reorder:
                num_concatenated_dims = num_frames * params['class_capsules']['caps_dim']
            else:
                num_concatenated_dims = num_frames * params['class_capsules']['caps_dim'] * \
                    params['class_capsules']['num_caps']
            # Either it's ordered correctly or not.
            self.ordering_head = nn.Linear(num_concatenated_dims, 1)

        self.get_mnist_head = mnist_classifier_head
        self.mnist_classifier_strategy = mnist_classifier_strategy
        if mnist_classifier_head:
            if mnist_classifier_strategy in ['pose', 'presence-pose']:
                num_params = params['class_capsules']['caps_dim'] * params['class_capsules']['num_caps']
            elif mnist_classifier_strategy == 'presence':
                num_params = params['class_capsules']['num_caps']
            self.mnist_classifier_head = nn.Linear(num_params, 10)

    def get_presence(self, pre_caps, final_pose):
        if 'squash' not in self.presence_loss_type and 'l2norm' not in self.presence_loss_type:
            presence_input = pre_caps.view(pre_caps.shape[0], -1)
            logits = self.presence_prob_head(presence_input)

        if self.presence_loss_type in [
                'sigmoid_l1', 'sigmoid_only', 'sigmoid_prior_sparsity',
                'sigmoid_prior_sparsity_fix',
                'sigmoid_prior_sparsity_example', 'sigmoid_within_entropy',
                'sigmoid_within_between_entropy', 'sigmoid_l1_between',
                'sigmoid_prior_sparsity_example_between_entropy',
                'sigmoid_prior_sparsity_between_entropy',
                'sigmoid_prior_sparsity_between_entropy_fix',
                'sigmoid_prior_sparsity_within_between_entropy_fix',
                'sigmoid_cossim', 'sigmoid_cossim_within_entropy',
                'sigmoid_hinge_presence'
        ]:
            logits *= self.presence_temperature
            # NOTE: we add noise here in order to try and spike it.
            rand_noise = torch.FloatTensor(logits.size()).uniform_(-2, 2).to(logits.device)
            logits += rand_noise
            logits = torch.sigmoid(logits)
        elif self.presence_loss_type in [
                'sigmoid_prior_sparsity_fix_nospike',
        ]:
            # NOTE: loook ma no noise!
            logits *= self.presence_temperature
            logits = torch.sigmoid(logits)
        elif self.presence_loss_type in [
                'softmax', 'softmax_prior_sparsity_example',
                'softmax_within_between_entropy'
        ]:
            logits *= self.presence_temperature
            # NOTE: we add noise here in order to try and spike it.
            rand_noise = torch.FloatTensor(logits.size()).uniform_(-2, 2).to(logits.device)
            logits += rand_noise
            logits = F.softmax(logits, 1)
        elif self.presence_loss_type in ['softmax_nonoise']:
            logits *= self.presence_temperature
            logits = F.softmax(logits, 1)
        elif 'squash' in self.presence_loss_type:
            # final_pose is [bs * num_images, num_capsules, capsule_dim]
            # We squash that here by doing
            # v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||
            # where s_j is the capsule dims. At the end, we now have a squashed
            # capsule dim, which we then take the l2_norm, which should be <=1.
            pose_normalized = final_pose / final_pose.norm(dim=2, keepdim=True)
            pose_norm_sq = final_pose.norm(dim=2, keepdim=True) ** 2
            logits = pose_norm_sq / (1 + pose_norm_sq) * pose_normalized
            logits = logits.norm(dim=2)
        elif 'l2norm' in self.presence_loss_type:
            logits = final_pose.norm(dim=2)
        else:
            raise
        return logits

    def get_ordering(self, x):
        return self.ordering_head(x)

    def forward(self, x, lbl_1=None, lbl_2=None, return_embedding=False,
                flatten=False, return_mnist_head=False, return_ordering_selection=None):
        #### Forward Pass
        ## Backbone (before capsule)
        # NOTE: pre_caps is [36, 128, 32, 32], or
        # [batch_size*num_images, backone['output_dim'],
        #  backbone['output_image_size], backbone['output_image_size']]
        # Lulz, this is 32x the input shape for mnist of [36, 1, 64, 64].
        print('X ', x.shape)

        # config=20ccgray --> precaps is [bs, backbone.output_dim=128, backbone.out_img_size=32, out_img_size=32]
        c = self.pre_caps(x)
        print('precaps', c.shape)
        pre_caps_res = c

        if self.do_capsule_computation:
            ## Primary Capsule Layer (a single CNN)
            # config=2ccgray --> u is [bs, something=576, primary_capsules.out_img_size, primary_capsules.out_img_size]
            u = self.pc_layer(c)
            print('u: ', u.shape)
            u = u.permute(0, 2, 3, 1)
            u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim,
                       self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
            u = u.permute(0, 3, 1, 2, 4)  # 100, 32, 14, 14, 16
            init_capsule_value = self.nonlinear_act(u)  #capsule_utils.squash(u)

            ## Main Capsule Layers
            # concurrent routing
            if not self.sequential_routing:
                # first iteration
                # perform initilialization for the capsule values as single forward passing
                capsule_values, _val = [init_capsule_value], init_capsule_value
                for i in range(len(self.capsule_layers)):
                    print('init capsule %d' % i, _val.shape)
                    # It's the CapsuleFC that knows to do this...
                    print(self.capsule_layers[i])
                    # For init capsule 0, we have [16, 16, 15, 15, 36].
                    # For init capsule 1, we have [16, 16, 13, 13, 36].
                    # For init capsule 2, we have [16, 10, 36].
                    # This is because capsule 0 is getting the output of u
                    # and then capsule 1 is getting the output of the Conv capsule
                    # and then capsule 2 is gettign the output of th FC capsule.
                    # X is [16, 1, 64, 64], which corresponds to [24, 3, 2048]
                    # precaps is [16, 128, 32, 32], compared to [24, 128, 1024].
                    # u is [16, 576, 15, 15], compared to [24, 512, 511].
                    # after permute, u is [16, 15, 15, 576] compared to [24, 511, 512]
                    # then here it's [16, 15, 15, 16, 36] and permuted --> [16, 16, 15, 15, 36].
                    # there it's [24, 511, 32, 16] and permuted --> [24, 32, 511, 16]
                    # ... which still appears correct.

                    # When this gets to init capsule 2, whic is an FC Presence,
                    # it's already of size [bs, nc, ncd]. For the conv ones, it's
                    # bigger. So .. that's a thing.
                    _val = self.capsule_layers[i].forward(_val, 0)
                    # Yeahhhh, so post capsule 1 is [16, 10, 36]... somehow it knows?
                    print('post capsule %d' % i, _val.shape)
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
            # sequential routing
            else:
                capsule_values, _val = [init_capsule_value], init_capsule_value
                for i in range(len(self.capsule_layers)):
                    # first iteration
                    __val = self.capsule_layers[i].forward(_val, 0)
                    # second to t iterations
                    # perform the routing between capsule layers
                    for n in range(self.num_routing - 1):
                        __val = self.capsule_layers[i].forward(_val, n, __val)
                    _val = __val
                    capsule_values.append(_val)

            ## After Capsule
            out = capsule_values[-1]
        else:
            out = self.dummy_pose.cuda(x.device)
            out = out[None, :, :].repeat(x.shape[0], 1, 1)

        pose = out
        if return_embedding:
            out = pose
            if flatten:
                ret = out.view(out.size(0), -1)
            else:
                ret = out
            return ret
        elif self.is_discriminating_model:
            ret = self.final_fc(pose).squeeze()
            return ret
        elif return_mnist_head:
            pose_shape = pose.shape
            if self.use_presence_probs:
                presence_probs = self.get_presence(pre_caps_res, pose.clone())
                if self.mnist_classifier_strategy == 'pose':
                    out = pose.view(pose_shape[0], -1)
                    out = self.mnist_classifier_head(out)
                elif self.mnist_classifier_strategy == 'presence-pose':
                    out = pose.clone() * presence_probs[:, :, None]
                    out = out.view(pose_shape[0], -1)
                    out = self.mnist_classifier_head(out)
                elif self.mnist_classifier_strategy == 'presence':
                    out = self.mnist_classifier_head(presence_probs)
                return out, pose, presence_probs
            else:
                out = pose.view(pose_shape[0], -1)
                out = self.mnist_classifier_head(out)
                return out, pose
        elif self.use_presence_probs:
            presence_probs = self.get_presence(pre_caps_res, pose.clone())
            if self.do_discriminative_probs:
                flattened = presence_probs.view(presence_probs.shape[0], -1)
                out = self.final_fc_probs(flattened)
                return pose, presence_probs, out
            # NOTE: moved get_nceprobs_selective_reorder_loss to here
            elif return_ordering_selection is not None:
                if return_ordering_selection == 'ncelinear_maxfirst':
                    stats = {}
                    num_images = 3
                    real_batch_size = pose.shape[0] // 3

                    pose = pose.view(real_batch_size, num_images, *pose.shape[1:])
                    presence_probs_image = presence_probs.view(
                        real_batch_size, num_images, -1)

                    presence_probs_image_detached = presence_probs_image.detach()
                    max_indices = torch.argmax(presence_probs_image_detached[:, 0], dim=1)
                    # shape of pose is [bs, nimg, ncaps, ndim], shape of indices is [bs].
                    # Now we want to get the nth index from max_indices from the nth batch
                    # entry and then combine them again.
                    selected_capsules = torch.stack(
                        [pose[num, :, index] for num, index in enumerate(max_indices)]
                    )
                    max_unique, max_counts = torch.unique(max_indices, return_counts=True)
                    max_counts = max_counts.float()
                    max_unique_counts, _ = torch.max(max_counts, 0)
                    mean_unique_counts = torch.mean(max_counts)
                    stats.update({
                        'max_count_maximally_activate_capsule': max_unique_counts.item(),
                        'mean_count_maximally_activate_capsule': mean_unique_counts.item(),
                    })

                    # ordering = model(selected_capsules.view(batch_size, -1), return_ordering=True).squeeze(-1)
                    ordering = self.ordering_head(selected_capsules.view(real_batch_size, -1)).squeeze(-1)

                    # Get the capsule distance stats.
                    sim_12 = torch.dist(selected_capsules[:, 0, :], selected_capsules[:, 1, :])
                    sim_23 = torch.dist(selected_capsules[:, 1, :], selected_capsules[:, 2, :])
                    sim_13 = torch.dist(selected_capsules[:, 0, :], selected_capsules[:, 2, :])
                    stats.update({
                        'capsule_12_sim': sim_12.item(),
                        'capsule_23_sim': sim_23.item(),
                        'capsule_13_sim': sim_13.item(),
                    })

                    return pose, presence_probs, ordering, stats
                else:
                    raise
            else:
                return pose, presence_probs
        else:
            # return pose, presence, object_
            return pose


class ProbsTest(nn.Module):
    def __init__(self, image_size, num_caps, temperature=1., use_noise=True):
        super(ProbsTest, self).__init__()
        self.linear = nn.Linear(image_size**2, num_caps)
        self.temperature = temperature
        self.use_noise = use_noise

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
        out *= self.temperature
        if self.use_noise:
            rand_noise = torch.FloatTensor(out.size()).uniform_(-2, 2).to(out.device)
            out += rand_noise
        out = torch.sigmoid(out)
        return out


class BackboneModel(nn.Module):

    def __init__(self,
                 params,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True,
                 mnist_classifier_head=False,
                 mnist_classifier_strategy='pose',
                 num_frames=4,
                 is_discriminating_model=False, # Use True if original model.
                 use_presence_probs=False,
                 presence_temperature=1.0,
                 presence_loss_type='sigmoid_l1',
                 do_capsule_computation=True,
                 do_discriminative_probs=False
    ):
        super(BackboneModel, self).__init__()

        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing  # >3 may cause slow converging

        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(
                params['backbone']['input_dim'],
                params['backbone']['output_dim'],
                params['backbone']['kernel_size'],
                params['backbone']['stride'],
                params['backbone']['padding'])
        elif backbone == 'resnet':
            self.pre_caps = layers.resnet_backbone(
                params['backbone']['input_dim'],
                params['backbone']['output_dim'],
                params['backbone']['stride'])

        input_dim = params['backbone']['output_dim']
        input_dim *= int(params['backbone']['inp_img_size']/2)**2
        output_dim = 10
        self.fc_head = nn.Linear(input_dim, output_dim)

        self.num_class_capsules = params['class_capsules']['num_caps']
        self.is_discriminating_model = is_discriminating_model
        self.use_presence_probs = use_presence_probs
        self.presence_temperature = presence_temperature
        self.presence_loss_type = presence_loss_type

    def forward(self, x, lbl_1=None, lbl_2=None, return_embedding=False,
                flatten=False, return_mnist_head=False):
        #### Forward Pass
        ## Backbone (before capsule)
        # NOTE: pre_caps is [36, 128, 32, 32], or
        # [batch_size*num_images, backone['output_dim'],
        #  backbone['output_image_size], backbone['output_image_size']]
        # Lulz, this is 32x the input shape for mnist of [36, 1, 64, 64].
        c = self.pre_caps(x)
        c = c.view(c.shape[0], -1)
        out = self.fc_head(c)
        return out


def get_bce_loss(model, images, labels):
    images = images[:, 0, :, :, :]
    output = model(images, return_embedding=False)
    loss = nn.BCEWithLogitsLoss()(output, labels)
    predicted = (torch.sigmoid(output) > 0.5).type(output.dtype)
    num_total = labels.size(0)
    num_targets = labels.eq(1).sum().item()
    correct = predicted.eq(labels).all(1).sum().item()
    true_positive_count = (predicted.eq(labels) & labels.eq(1)).sum().item()
    stats = {
        'true_pos': true_positive_count,
        'num_targets': num_targets
    }
    return loss, stats


def get_triplet_loss(model, images):
    inputs_v = model(inputs, return_embedding=True, flatten=False)
    positive_v = model(positive, return_embedding=True, flatten=False)
    # TODO: if batch size is odd num then it won't work
    negative_v = positive_v.flip(0)

    # Compute distance
    with torch.no_grad():
        positive_distance = float(torch.dist(inputs_v, positive_v, 2).item())
        negative_distance = float(torch.dist(inputs_v, negative_v, 2).item())

    stats = {'positive_distance': positive_distance,
             'negative_distance': negative_distance}

    return loss, stats


def get_nce_loss(model, images, args):
    positive_frame_num = args.nce_positive_frame_num
    use_random_anchor_frame = args.use_random_anchor_frame

    if use_random_anchor_frame:
        # NOTE: not implemented.
        raise
    else:
        anchor_frame = 0

    anchor = images[:, anchor_frame]
    anchor = model(anchor, return_embedding=True, flatten=False)
    other = images[:, anchor_frame + positive_frame_num]
    other = model(other, return_embedding=True, flatten=False)

    # Anchor / Positive should now be [bs, num_capsules, num_dims].
    # We get the similarity of these via
    # sim(X, Y) = temp * anchor * other / (||anchor|| * ||other||)
    # ... Do we want this to happen over capsules or over flattened?
    # I think it should happen over capsules. So the rescaling would occur
    # over each capsule's dims rather than all of it in total.
    batch_size = anchor.size(0)
    anchor_normalized = anchor / anchor.norm(dim=2, keepdim=True)
    anchor_normalized = anchor_normalized.view(batch_size, 1, -1)
    other_normalized = other / other.norm(dim=2, keepdim=True)
    other_normalized = other_normalized.view(1, batch_size, -1)

    # similarity will be [bs, bs, num_capsules * num_dims] after this.
    similarity = anchor_normalized * other_normalized
    # now multiply by the temperature.
    similarity *= args.nce_temperature
    # and then sum to get the dot product (cosign similarity).
    # this is [bs, bs]. the positive samples are on the diagonal.
    similarity = similarity.sum(2)

    # the diagonal has the positive similarity = log(exp(sim(x, y)))
    identity = torch.eye(batch_size).to(similarity.device)
    positive_similarity = (similarity * identity).sum(1)

    # we get the total similarity by taking the logsumexp of similarity.
    log_sum_total = torch.logsumexp(similarity, dim=1)

    diff = positive_similarity - log_sum_total
    nce = -diff.mean()

    total_similarity = similarity.sum(1)
    negative_similarity = (total_similarity - positive_similarity).mean().item() / (batch_size - 1)
    positive_similarity = positive_similarity.mean().item()
    stats = {
        'pos_sim': positive_similarity,
        'neg_sim': negative_similarity
    }
    return nce, stats


def get_xent_loss(model, images, labels):
    output = model(images, return_embedding=False)
    loss = F.cross_entropy(output, labels)
    predictions = torch.argmax(output, dim=1)
    accuracy = (predictions == labels).float().mean().item()
    stats = {
        'accuracy': accuracy,
    }
    return loss, stats


def get_reorder_loss(model, images, device, args, labels=None):
    # print('now shape: ', images_shape)
    # path = '/misc/kcgscratch1/ChoGroup/resnick/up%d.vid%d.png'
    # for i in range(3):
    #     arr = images[0][i].numpy()
    #     arr = (255 * arr).astype(np.uint8)
    #     print(arr.shape, arr.dtype)
    #     img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
    #     img.save(path % (int(use_positive), i))

    batch_size, num_images = images.shape[:2]
    # Change view so that we can put everything through the model at once.
    images = images.view(batch_size * num_images, *images.shape[2:])
    pose = model(images)
    pose = pose.view(batch_size, num_images, *pose.shape[1:])

    # presence = presence.view(batch_size, num_images, *presence.shape[1:])
    # object_ = object_.view(batch_size, num_images, *object_.shape[1:])
    # We now want object_presence to be equal across frames and
    # object_pose_presence to be indicative of the ordering.
    # The objects_ right now are Concatenate the objects so that we can compare them.
    # [4,3,10,16] ... [4,3,10,1] ... [4,3,,10,36]
    # object_presence = object_ * presence
    # object_pose_presence = object_presence * pose

    # Get that the image representations should be the same.
    # object_presence = object_presence.view(batch_size, num_images, -1)
    # Going with cosine similarity here.
    # transposed_object_presence = object_presence.permute(0, 2, 1)
    # rolled_object_presence[:, :, 0] = transposed_object_presence[:, :, -1]
    # rolled_object_presence = torch.cat(
    #     (transposed_object_presence[:, :, -1:],
    #      transposed_object_presence[:, :, :-1]),
    #     dim=2
    # )

    # cosine_sim = F.cosine_similarity(
    #     transposed_object_presence, rolled_object_presence, dim=1)

    # Get the ordering.
    # flattened = object_pose_presence.view(batch_size, -1)
    flattened = pose.view(batch_size, -1)

    # TODO: Change this in order to get it working with DDP
    ordering = model.module.ordering_head(flattened).squeeze(-1)

    # loss_objects = -cosine_sim.sum(1)
    # loss_sparsity = presence.sum((1, 2))
    # loss_objects = loss_objects.mean()
    # loss_sparsity = loss_sparsity.mean()
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


def get_reorder_loss2(model, images, device, args, labels=None):
    """
    Difference between this and the above is that this uses 4 frames in the
    sequence, not 3. But also it's made much easier.
    """
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
        # frames (b, c, d, e), (a, b, c, d), (e, d, c, b), or (d, c, b, a).
        if random.random() > 0.5:
            selection = sample[:4]
        else:
            selection = sample[1:]
    else:
        # (a, b, c, d, e)
        # (b, c, a, d), (a, c, d, b), (b, d, e, c),
        # (c, a, b, d), (a, d, b, c), (b, e, c, d)
        if random.random() > .84:
            selection = [sample[1], sample[2], sample[0], sample[3]]
        elif random.random() > .68:
            selection = [sample[0], sample[2], sample[3], sample[1]]
        elif random.random() > .52:
            selection = [sample[1], sample[3], sample[4], sample[2]]
        elif random.random() > .36:
            selection = [sample[2], sample[0], sample[1], sample[3]]
        elif random.random() > .20:
            selection = [sample[0], sample[3], sample[1], sample[2]]
        else:
            selection = [sample[1], sample[4], sample[2], sample[3]]

    images = images[:, selection]

    # path = '/misc/kcgscratch1/ChoGroup/resnick/mm2.vid%d.frame%d.after%d.png'
    # for batch_num in range(len(images)):
    #     image = images[batch_num].numpy()
    #     label_ = labels[batch_num].numpy()
    #     for num in range(len(image)):
    #         arr = image[num]
    #         arr = (255 * arr).astype(np.uint8)
    #         img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
    #         img.save(path % (label_, num, int(use_positive)))

    images_shape = images.shape
    # print('now shape: ', images_shape)
    # path = '/misc/kcgscratch1/ChoGroup/resnick/up%d.vid%d.png'
    # for i in range(3):
    #     arr = images[0][i].numpy()
    #     arr = (255 * arr).astype(np.uint8)
    #     print(arr.shape, arr.dtype)
    #     img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
    #     img.save(path % (int(use_positive), i))

    batch_size, num_images = images.shape[:2]
    images = images.to(device)
    labels = torch.tensor([use_positive]*batch_size).type(torch.FloatTensor).to(images.device)

    # Change view so that we can put everything through the model at once.
    images = images.view(batch_size * num_images, *images.shape[2:])
    pose = model(images)
    pose = pose.view(batch_size, num_images, *pose.shape[1:])

    # presence = presence.view(batch_size, num_images, *presence.shape[1:])
    # object_ = object_.view(batch_size, num_images, *object_.shape[1:])
    # We now want object_presence to be equal across frames and
    # object_pose_presence to be indicative of the ordering.
    # The objects_ right now are Concatenate the objects so that we can compare them.
    # [4,3,10,16] ... [4,3,10,1] ... [4,3,,10,36]
    # object_presence = object_ * presence
    # object_pose_presence = object_presence * pose

    # Get that the image representations should be the same.
    # object_presence = object_presence.view(batch_size, num_images, -1)
    # Going with cosine similarity here.
    # transposed_object_presence = object_presence.permute(0, 2, 1)
    # rolled_object_presence[:, :, 0] = transposed_object_presence[:, :, -1]
    # rolled_object_presence = torch.cat(
    #     (transposed_object_presence[:, :, -1:],
    #      transposed_object_presence[:, :, :-1]),
    #     dim=2
    # )

    # cosine_sim = F.cosine_similarity(
    #     transposed_object_presence, rolled_object_presence, dim=1)

    # Get the ordering.
    # flattened = object_pose_presence.view(batch_size, -1)
    flattened = pose.view(batch_size, -1)
    ordering = model.ordering_head(flattened).squeeze(-1)

    # loss_objects = -cosine_sim.sum(1)
    # loss_sparsity = presence.sum((1, 2))
    # loss_objects = loss_objects.mean()
    # loss_sparsity = loss_sparsity.mean()
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


def get_triangle_loss(self, images, device, args):
    batch_size, num_images = images.shape[:2]

    # Change view so that we can put everything through the model at once.
    images = images.view(batch_size * num_images, *images.shape[2:])
    pose = self(images)
    # With 2channel, this is torch.Size([12, 3, 2, 36]).
    # With 1channel, it is torch.Size([12, 5, 2, 36])
    # NOTE: This is ... [bs, ni, 2] when doing the 1channel... is that right?
    pose = pose.view(batch_size, num_images, *pose.shape[1:])

    sim_12 = torch.dist(pose[:, 0, :], pose[:, 1, :])
    sim_23 = torch.dist(pose[:, 1, :], pose[:, 2, :])
    sim_13 = torch.dist(pose[:, 0, :], pose[:, 2, :])

    segment_13 = pose[:, 2, :] - pose[:, 0, :]
    segment_12 = pose[:, 1, :] - pose[:, 0, :]
    cosine_sim_13_12 = F.cosine_similarity(segment_13, segment_12).mean()

    if args.criterion == 'triangle_margin2_angle':
        # Here, we optimize the angle between them. We will pair it down
        # below with the margin_loss_12 + margin_loss_23.
        loss = -cosine_sim_13_12
    else:
        # Here, we optimize the triangle expression. This going to zero
        # should surrogate optimize the angle above.
        loss = sim_12 + sim_23 - args.triangle_lambda * sim_13

    stats = {
        'frame_12_sim': sim_12.item(),
        'frame_23_sim': sim_23.item(),
        'frame_13_sim': sim_13.item(),
        'triangle_margin': (sim_13 - sim_12 - sim_23).item(),
        'cosine_sim_13_12': cosine_sim_13_12.item()
    }

    if args.criterion == 'triangle_cos':
        # We want to minimize cosine similarity. Maximizing it would mean
        # that the angle between them was theta.
        cos_sim_13 = F.cosine_similarity(pose[:, 0, :], pose[:, 2, :]).mean()
        stats['frame_13_cossim'] = cos_sim_13.item()
        loss += args.triangle_cos_lambda * cos_sim_13
    elif args.criterion == 'triangle_margin':
        # We want to try to get the distance between pose of frames 1 and 3
        # to be at least the margin size.
        margin_loss = torch.norm(segment_13, dim=1)
        margin_loss = args.margin_gamma - margin_loss
        margin_loss = F.relu(margin_loss).sum()
        loss += margin_loss * args.triangle_margin_lambda
        stats['margin_loss'] = margin_loss.item()
    elif args.criterion in ['triangle_margin2', 'triangle_margin2_angle']:
        # Here we put the margin_loss on segmnet_12 and semgnet_23 instead.
        segment_23 = pose[:, 2, :] - pose[:, 1, :]
        margin_loss_23 = torch.norm(segment_23, dim=1)
        margin_loss_23 = args.margin_gamma2 - margin_loss_23
        margin_loss_23 = F.relu(margin_loss_23).sum()
        loss += margin_loss_23 * args.triangle_margin_lambda

        margin_loss_12 = torch.norm(segment_12, dim=1)
        margin_loss_12 = args.margin_gamma2 - margin_loss_12
        margin_loss_12 = F.relu(margin_loss_12).sum()
        loss += margin_loss_12 * args.triangle_margin_lambda

        stats['margin_loss_23'] = margin_loss_23.item()
        stats['margin_loss_12'] = margin_loss_12.item()
        stats['margin_loss'] = stats['margin_loss_23'] + stats['margin_loss_12']

    return loss, stats


def get_discriminative_probs(model, images, labels, device, epoch, args):
    use_two_images = not args.use_hinge_loss and not args.use_angle_loss
    batch_size, num_images = images.shape[:2]

    # Change view so that we can put everything through the model at once.
    images = images.view(batch_size * num_images, *images.shape[2:])
    pose, presence_probs, class_probs = model(images)
    if 'nomul' not in args.presence_loss_type:
        pose *= presence_probs[:, :, None]

    pose = pose.view(batch_size, num_images, *pose.shape[1:])
    presence_probs_image = presence_probs.view(batch_size, num_images, -1)
    class_probs = class_probs.view(batch_size, num_images, -1)
    # Take the mean of the probs over the images, i.e. we want them all to end
    # up as the same thing. And that's the loss.
    class_probs = class_probs.mean(1)

    loss = F.cross_entropy(class_probs, labels)
    predictions = torch.argmax(class_probs, dim=1)
    accuracy = (predictions == labels).float().mean().item()
    stats = {
        'accuracy': accuracy,
    }
    return loss, stats


def get_triangle_nce_loss(model, images, device, epoch, args,
                          num_class_capsules=10):
    """Get linearizing loss AND the NCE loss.

    We optimize the NCE(f1_i, f3_j) so that the model learns to distinguish the
    objects here from other objects.

    However, this will have the tendency to push the pose to be the same. We
    don't want that because we want pose to have meaning related to affine
    transformations.

    That meaning can come from S&L or it can come from linearization. Here, we
    choose to do linearization.

    That means optimizing the angle between (f1, f2) and (f1, f3) to be zero
    so that the pose is linear. This can lead to degenerate solutions though
    because the easiest thing to do then is to make fk_i equal for all k. So we
    further also add a margin loss between (f1, f2) and (f1, f3).
    """
    use_two_images = not args.use_hinge_loss and not args.use_angle_loss
    batch_size, num_images = images.shape[:2]

    # Change view so that we can put everything through the model at once.
    images = images.view(batch_size * num_images, *images.shape[2:])
    if args.use_presence_probs:
        pose, presence_probs = model(images)
        if 'nomul' not in args.presence_loss_type:
            pose *= presence_probs[:, :, None]
    else:
        pose = model(images)
    pose = pose.view(batch_size, num_images, *pose.shape[1:])
    presence_probs_image = presence_probs.view(
        batch_size, num_images, -1)

    stats = {}

    sim_12 = torch.dist(pose[:, 0, :], pose[:, 1, :])
    segment_12 = pose[:, 1, :] - pose[:, 0, :]
    stats['frame_12_sim'] = sim_12.item()

    if not use_two_images:
        sim_23 = torch.dist(pose[:, 1, :], pose[:, 2, :])
        sim_13 = torch.dist(pose[:, 0, :], pose[:, 2, :])
        segment_13 = pose[:, 2, :] - pose[:, 0, :]
        segment_23 = pose[:, 2, :] - pose[:, 1, :]
        stats.update({
            'frame_23_sim': sim_23.item(),
            'frame_13_sim': sim_13.item(),
            'triangle_margin': (sim_13 - sim_12 - sim_23).item(),
        })

    loss = 0.

    # pose[:, 0, :] is [bs, num_capsules, num_dim], presence_probs_image is [bs, ni, num_capsules]

    if args.use_nce_probs:
        nce, stats_ = _do_simclr_nce(args.nce_presence_temperature,
                                     presence_probs_image)
        loss += nce * args.nce_presence_lambda
        stats.update(stats_)

    if args.use_angle_loss:
        # Optimize the angle between (f1, f2) and (f1, f3).
        cosine_sim_13_12 = F.cosine_similarity(segment_13, segment_12).mean()
        loss += -cosine_sim_13_12 * args.triangle_cos_lambda
        stats.update({
            'cosine_sim_13_12': cosine_sim_13_12.item(),
        })

    if args.use_hinge_loss:
        # Get the margin_loss on segment_12 and semgent_23.
        margin_loss_23 = torch.norm(segment_23, dim=1)
        margin_loss_23 = args.margin_gamma2 - margin_loss_23
        margin_loss_23 = F.relu(margin_loss_23).sum()
        margin_loss_12 = torch.norm(segment_12, dim=1)
        margin_loss_12 = args.margin_gamma2 - margin_loss_12
        margin_loss_12 = F.relu(margin_loss_12).sum()
        loss += margin_loss_23 * args.triangle_margin_lambda
        loss += margin_loss_12 * args.triangle_margin_lambda
        stats.update({
            'margin_loss_23': margin_loss_23.item(),
            'margin_loss_12': margin_loss_12.item(),
        })
        stats['margin_loss'] = stats['margin_loss_23'] + stats['margin_loss_12']

    # Get the NCE loss.
    if args.use_nce_loss:
        anchor = pose[:, 0, :]
        if use_two_images:
            other = pose[:, 1, :]
        else:
            other = pose[:, 2, :]

        anchor_normalized = anchor / (anchor.norm(dim=2, keepdim=True) + 1e-6)
        anchor_normalized = anchor_normalized.view(batch_size, 1, -1)
        other_normalized = other / (other.norm(dim=2, keepdim=True) + 1e-6)
        other_normalized = other_normalized.view(1, batch_size, -1)
        # similarity will be [bs, bs, num_capsules * num_dims] after this.
        similarity = anchor_normalized * other_normalized
        # now multiply by the temperature.
        similarity *= args.nce_temperature
        # and then sum to get the dot product (cosign similarity).
        # this is [bs, bs]. the positive samples are on the diagonal.
        similarity = similarity.sum(2)
        # the diagonal has the positive similarity = log(exp(sim(x, y)))
        identity = torch.eye(batch_size).to(similarity.device)
        positive_similarity = (similarity * identity).sum(1)

        # we get the total similarity by taking the logsumexp of similarity.
        log_sum_total = torch.logsumexp(similarity, dim=1)
        diff = positive_similarity - log_sum_total
        nce = -diff.mean()

        total_similarity = similarity.sum(1)
        negative_similarity = (total_similarity - positive_similarity).mean().item() / (batch_size - 1)
        positive_similarity = positive_similarity.mean().item()

        loss += nce * args.nce_lambda
        stats.update({
            'pos_sim': positive_similarity,
            'neg_sim': negative_similarity,
            'nce': nce.item()
        })

    if args.use_presence_probs:
        # In addition to the loss, we also want to get stats that include the
        # sum per capsule as well as how close are the sums across images of
        # the same video.

        # NOTE: Omgahd, none of these have been correct as of May 4th. Ugh.
        # The below is fixed now. It was using norm instead of sum before.
        presence_probs_first = presence_probs_image[:, 0]

        presence_probs_sum1 = presence_probs_first.sum(dim=1, keepdim=True) + 1e-8
        within_presence = presence_probs_first / presence_probs_sum1 + 1e-8
        within_entropy = -within_presence * torch.log(within_presence) / np.log(2)
        # This is correct with a shape of 24.
        within_entropy = within_entropy.sum(1)
        within_entropy = within_entropy.mean()

        # NOTE: This ALSO isn't correct. It's a different value than what we
        # want. What we want is to have the sums of the capsules be pretty much
        # the same, i.e. for them all to be used. So we want the entropy on top
        # of the sum of these numbers to be large.
        # (It's now correct --> May 5th)
        presence_probs_sum0 = presence_probs_first.sum(dim=0) + 1e-8
        presence_probs_sum0_distr = presence_probs_sum0 / presence_probs_sum0.sum(0)
        between_entropy = -presence_probs_sum0_distr * torch.log(presence_probs_sum0_distr) / np.log(2)
        between_entropy = between_entropy.mean(0)

        stats['within_entropy'] = within_entropy.item()
        stats['between_entropy'] = between_entropy.item()
        max_values, _ = torch.max(presence_probs_first, 1)
        min_values, _ = torch.min(presence_probs_first, 1)
        stats['mean_max_prob'] = max_values.mean().item()
        stats['mean_min_prob'] = min_values.mean().item()
        stats['mean_prob'] = presence_probs_first.mean().item()

        if args.presence_loss_type == 'sigmoid_only':
            pass
        elif args.presence_loss_type == 'sigmoid_l1':
            # NOTE: Using this presence loss results in the model always using
            # the same subset of capsules, regardless of input.
            presence_loss = presence_probs.sum(1).mean()
            loss += presence_loss * args.lambda_sparse_presence
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type == 'sigmoid_l1_between':
            # NOTE: Using this presence loss results in the model always using
            # the same subset of capsules, regardless of input.
            presence_loss = presence_probs.sum(1).mean()
            loss += presence_loss * args.lambda_sparse_presence
            loss -= between_entropy * args.lambda_between_entropy
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type == 'sigmoid_prior_sparsity_example':
            # Ok, yeah if we keep this going, then it gets everything good
            # except the example_presence_loss only goes to 1.0. It ends up
            # using two capsules instead of one. ... Maybe that's because it
            # needs to, i.e. it doesn't have enough in just one capsule?
            # That suggests increasng hte number of capsules.
            # BUT it also collapses to just using the same two. So that's not
            # useful. We still need something to push all the capsules to get
            # used.

            # The example_presence_loss says that the sum of probabilities
            # should sum to the number of capsules / number of classes, which
            # is 1 for MovingMNist with 10 capsules. With just this in place,
            # the model could do [0.1]*10 but what we really want is more like
            # [1.] + [0]*9.
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type == 'sigmoid_prior_sparsity':
            # This is doing the above example_presence_loss, but then also
            # adding in a capsule_presence_loss, whose aim is to get each of
            # the capsules to be present for at least one of every output_class.
            # NOTE: The assumption there is that there is only one object in
            # each image. This would have to be changed for adding in more classes.
            # NOTE: This was not possible to do because the batch_size was incorrect for presence_probs to use!!!
            # See below for the fix (sigmoid_prior_sparsity_fix)
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            target_capsule_presence = batch_size * 1. / args.num_output_classes
            capsule_presence_sum = presence_probs.sum(0)
            capsule_presence_loss = ((capsule_presence_sum - target_capsule_presence)**2).mean()
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = capsule_presence_loss + example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            stats['capsule_presence_loss'] = capsule_presence_loss.item()
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type == 'sigmoid_prior_sparsity_between_entropy':
            # This is doing the above example_presence_loss, but then also
            # adding in a capsule_presence_loss, whose aim is to get each of
            # the capsules to be present for at least one of every output_class.
            # NOTE: The assumption there is that there is only one object in
            # each image. This would have to be changed for adding in more classes.
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            target_capsule_presence = batch_size * 1. / args.num_output_classes
            capsule_presence_sum = presence_probs.sum(0)
            capsule_presence_loss = ((capsule_presence_sum - target_capsule_presence)**2).mean()
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = capsule_presence_loss + example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            loss -= between_entropy * args.lambda_between_entropy
            stats['capsule_presence_loss'] = capsule_presence_loss.item()
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type == 'sigmoid_prior_sparsity_example_between_entropy':
            # Here, we use the example sparsity in order to get this to use only
            # a sparse number of capsules.
            # We then maximize between_entropy in order to get this to put weight
            # on all of the capsules at some point.
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            loss -= between_entropy * args.lambda_between_entropy
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type == 'sigmoid_within_entropy':
            # Here, we use entropy constraints. We normalize the probabilities
            # and we try to minimize the within-example entropy. Minimizing the
            # entropy would result in a sharp distribution.
            # I suspect that this would enable the model to use few of the
            # capsules and be fine. Let's see if that happens.
            # NOTE: So what this does is the model uses few of the capsules but
            # then doesn't spread them around. Everyone just uses the capsules.
            loss += within_entropy * args.lambda_within_entropy
        elif args.presence_loss_type == 'sigmoid_within_between_entropy':
            # Here, we keep the within_exampel entropy, but we add the between
            # example entropy. By maximizing the latter, we should end up with
            # at least some mass placed on every capsule.
            # This one has an issue where it's really volatile and otherwise
            # seems to act as either everyone gets weight or we only do
            # sigmoid_within_entropy.
            # NOTE: We might not have enough in the batch for this ...
            # TODO: I should redo these given the change to between_sparsity
            # being that of using only image one.
            loss += within_entropy * args.lambda_within_entropy
            loss -= between_entropy * args.lambda_between_entropy
        elif args.presence_loss_type == 'sigmoid_cossim':
            # presence_probs_image is [bs, num_images, num_capsules] and
            # represents the capsule probabilities per image. In this loss,
            # we maximize the cossim of [bs_k, 0] with [bs_k, 1] and minimize the
            # cossim of [bs_k, 0] with [bs_j, 0]
            same_capsule_cossim = F.cosine_similarity(
                presence_probs_image[:, 0], presence_probs_image[:, 1], dim=1
            ).mean()
            # rolled_presence_probs_image is a 1-permutation of presence_probs_image.
            # rolled_presence_probs_image = torch.cat(
            #     (presence_probs_image[-1:, :, :],
            #      presence_probs_image[:-1, :, :]),
            #     dim=0
            # )
            rolled_presence_probs_image = torch.roll(presence_probs_image, 1, 0)
            diff_capsule_cossim = F.cosine_similarity(
                presence_probs_image[:, 0], rolled_presence_probs_image[:, 0],
                dim=1
            ).mean()
            # loss += same_capsule_cossim * args.presence_samecos_lambda
            loss -= diff_capsule_cossim * args.presence_diffcos_lambda
            stats['same_capsule_cossim_loss'] = same_capsule_cossim.item()
            stats['diff_capsule_cossim_loss'] = diff_capsule_cossim.item()
        elif args.presence_loss_type == 'sigmoid_cossim_within_entropy':
            # presence_probs_image is [bs, num_images, num_capsules] and
            # represents the capsule probabilities per image. In this loss,
            # we maximize the cossim of [bs_k, 0] with [bs_k, 1] and minimize the
            # cossim of [bs_k, 0] with [bs_j, 0]
            same_capsule_cossim = F.cosine_similarity(
                presence_probs_image[:, 0], presence_probs_image[:, 1], dim=1
            ).mean()
            # rolled_presence_probs_image is a 1-permutation of presence_probs_image.
            # rolled_presence_probs_image = torch.cat(
            #     (presence_probs_image[-1:, :, :],
            #      presence_probs_image[:-1, :, :]),
            #     dim=0
            # )
            rolled_presence_probs_image = torch.roll(presence_probs_image, 1, 0)
            diff_capsule_cossim = F.cosine_similarity(
                presence_probs_image[:, 0], rolled_presence_probs_image[:, 0],
                dim=1
            ).mean()
            # loss += same_capsule_cossim * args.presence_samecos_lambda
            loss -= diff_capsule_cossim * args.presence_diffcos_lambda
            loss += within_entropy * args.lambda_within_entropy
            stats['same_capsule_cossim_loss'] = same_capsule_cossim.item()
            stats['diff_capsule_cossim_loss'] = diff_capsule_cossim.item()
        elif args.presence_loss_type == 'softmax':
            # This seems to result in everyone getting the same capsule. Let's
            # see what happens if we let it go longer.
            pass
        elif args.presence_loss_type == 'softmax_nonoise':
            # This also results in everyone getting the same capsule.
            pass
        elif args.presence_loss_type == 'softmax_within_between_entropy':
            loss += within_entropy * args.lambda_within_entropy
            loss -= between_entropy * args.lambda_between_entropy
        elif args.presence_loss_type in [
                'squash_prior_sparsity', 'squash_prior_sparsity_nomul',
                'squash_prior_sparsity_within_entropy',
                'squash_prior_sparsity_within_entropy_nomul',
        ]:
            # NOTE: These are incorrect as well.
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            target_capsule_presence = batch_size * 1. / args.num_output_classes
            capsule_presence_sum = presence_probs.sum(0)
            capsule_presence_loss = ((capsule_presence_sum - target_capsule_presence)**2).mean()
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = capsule_presence_loss + example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            stats['capsule_presence_loss'] = capsule_presence_loss.item()
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type in [
                'squash_prior_sparsity_within_entropy',
                'squash_prior_sparsity_within_entropy_nomul',
        ]:
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            target_capsule_presence = batch_size * 1. / args.num_output_classes
            capsule_presence_sum = presence_probs.sum(0)
            capsule_presence_loss = ((capsule_presence_sum - target_capsule_presence)**2).mean()
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = capsule_presence_loss + example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            stats['capsule_presence_loss'] = capsule_presence_loss.item()
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type in [
                'squash_within_between_entropy', 'squash_within_between_entropy_nomul'
        ]:
            loss += within_entropy * args.lambda_within_entropy
            loss -= between_entropy * args.lambda_between_entropy
        elif args.presence_loss_type in ['squash_cossim', 'squash_cossim_nomul']:
            # presence_probs_image is [bs, num_images, num_capsules] and
            # represents the capsule probabilities per image. In this loss,
            # we maximize the cossim of [bs_k, 0] with [bs_k, 1] and minimize the
            # cossim of [bs_k, 0] with [bs_j, 0]
            same_capsule_cossim = F.cosine_similarity(
                presence_probs_image[:, 0], presence_probs_image[:, 1], dim=1
            ).mean()
            # rolled_presence_probs_image is a 1-permutation of presence_probs_image.
            # rolled_presence_probs_image = torch.cat(
            #     (presence_probs_image[-1:, :, :],
            #      presence_probs_image[:-1, :, :]),
            #     dim=0
            # )
            rolled_presence_probs_image = torch.roll(presence_probs_image, 1, 0)
            diff_capsule_cossim = F.cosine_similarity(
                presence_probs_image[:, 0], rolled_presence_probs_image[:, 0],
                dim=1
            ).mean()
            # loss += same_capsule_cossim * args.presence_samecos_lambda
            loss -= diff_capsule_cossim * args.presence_diffcos_lambda
            stats['same_capsule_cossim_loss'] = same_capsule_cossim.item()
            stats['diff_capsule_cossim_loss'] = diff_capsule_cossim.item()
        elif args.presence_loss_type in [
                'squash_example_between', 'squash_example_between_nomul'
        ]:
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            loss -= between_entropy * args.lambda_between_entropy
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type in [
                'sigmoid_hinge_presence'
        ]:
            # This is saying ok look, we want each capsule to be on for some
            # percent of the batch on average. Say that there are C classes
            # and B batch entries. Then each capsule should be 1 for ~B/C of
            # the time. We tried doing L2 above re the capsule_presence, but
            # let's makign that a bit softer and doing a margin loss, so that
            # it's at LEAST that amount.
            # NOTE: My prediction is that this will make it so that each wants
            # to be responsible for at least blah of the
            target_capsule_presence = batch_size * 1. / args.num_output_classes
            capsule_presence_sum = presence_probs_first.sum(0)

            # The hinge now is on the capsule_presence_sum being greater than
            # the target_capsule_presence.
            hinge_capsule_presence = target_capsule_presence - capsule_presence_sum
            # If hinge_capsule_presence > 0, then the capsule_presence_sum is too small.
            # So we penalize it by saying that the loss is
            # F.relu(hinge_capsule_presence)
            hinge_capsule_presence_loss = F.relu(hinge_capsule_presence).mean()
            loss += hinge_capsule_presence_loss * args.hinge_presence_loss
            stats['presence_hinge_loss'] = hinge_capsule_presence_loss.item()
        elif args.presence_loss_type in [
                'sigmoid_prior_sparsity_fix', 'sigmoid_prior_sparsity_fix_nospike'
        ]:
            # This is fixing the sigmoid_prior_sparsity up above, which was
            # using the wrong target_capsule_presence gien the size of the size
            # of the presence_probs.
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            target_capsule_presence = batch_size * num_images * 1. / args.num_output_classes
            capsule_presence_sum = presence_probs.sum(0)
            capsule_presence_loss = ((capsule_presence_sum - target_capsule_presence)**2).mean()
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = capsule_presence_loss + example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            stats['capsule_presence_loss'] = capsule_presence_loss.item()
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type == 'sigmoid_prior_sparsity_between_entropy_fix':
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            target_capsule_presence = batch_size * num_images * 1. / args.num_output_classes
            capsule_presence_sum = presence_probs.sum(0)
            capsule_presence_loss = ((capsule_presence_sum - target_capsule_presence)**2).mean()
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = capsule_presence_loss + example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            loss -= between_entropy * args.lambda_between_entropy
            stats['capsule_presence_loss'] = capsule_presence_loss.item()
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type == 'sigmoid_prior_sparsity_within_between_entropy_fix':
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            target_capsule_presence = batch_size * num_images * 1. / args.num_output_classes
            capsule_presence_sum = presence_probs.sum(0)
            capsule_presence_loss = ((capsule_presence_sum - target_capsule_presence)**2).mean()
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = capsule_presence_loss + example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            loss += within_entropy * args.lambda_within_entropy
            loss -= between_entropy * args.lambda_between_entropy
            stats['capsule_presence_loss'] = capsule_presence_loss.item()
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type in [
                'squash_prior_sparsity_fix', 'squash_prior_sparsity_nomul_fix'
        ]:
            # NOTE: These are incorrect as well.
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            target_capsule_presence = batch_size * num_images * 1. / args.num_output_classes
            capsule_presence_sum = presence_probs.sum(0)
            capsule_presence_loss = ((capsule_presence_sum - target_capsule_presence)**2).mean()
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = capsule_presence_loss + example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            stats['capsule_presence_loss'] = capsule_presence_loss.item()
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type in [
                'squash_prior_sparsity_within_entropy_fix',
                'squash_prior_sparsity_within_entropy_nomul_fix',
        ]:
            # NOTE: These are incorrect as well.
            target_example_presence = model.module.num_class_capsules * 1. / args.num_output_classes
            target_capsule_presence = batch_size * num_images * 1. / args.num_output_classes
            capsule_presence_sum = presence_probs.sum(0)
            capsule_presence_loss = ((capsule_presence_sum - target_capsule_presence)**2).mean()
            example_presence_sum = presence_probs.sum(1)
            example_presence_loss = ((example_presence_sum - target_example_presence)**2).mean()
            presence_loss = capsule_presence_loss + example_presence_loss
            loss += presence_loss * args.lambda_sparse_presence
            loss += within_entropy * args.lambda_within_entropy
            stats['capsule_presence_loss'] = capsule_presence_loss.item()
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        else:
            raise

        mean_per_capsule = [value.item() for value in presence_probs.mean(0)]
        std_per_capsule = [value.item() for value in presence_probs.std(0)]

        l2_probs_12 = torch.norm(
            presence_probs_image[:, 0, :] - presence_probs_image[:, 1, :],
            dim=1).mean()
        stats.update({
            'l2_presence_probs_12': l2_probs_12.item(),
        })
        if not use_two_images:
            l2_probs_13 = torch.norm(
                presence_probs_image[:, 0, :] - presence_probs_image[:, 2, :],
                dim=1).mean()
            l2_probs_23 = torch.norm(
                presence_probs_image[:, 1, :] - presence_probs_image[:, 2, :],
                dim=1).mean()
            stats.update({
                'l2_presence_probs_13': l2_probs_13.item(),
                'l2_presence_probs_23': l2_probs_23.item(),
            })

        for num, item in enumerate(mean_per_capsule):
            stats['capsule_prob_mean_%d' % num] = item
        for num, item in enumerate(std_per_capsule):
            stats['capsule_prob_std_%d' % num] = item

    return loss, stats


def _do_simclr_nce(temperature, probs=None, anchor=None, other=None,
                   suffix='presence', selection_strategy='default',
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


def _do_our_nce(probs, args):
    batch_size = probs.shape[0]
    anchor = probs[:, 0]
    other = probs[:, 1]
    anchor_normalized = anchor / (anchor.norm(dim=1, keepdim=True) + 1e-6)
    anchor_normalized = anchor_normalized.view(batch_size, 1, -1)
    other_normalized = other / (other.norm(dim=1, keepdim=True) + 1e-6)
    other_normalized = other_normalized.view(1, batch_size, -1)
    # similarity will be [bs, bs, num_capsules * num_dims] after this.
    similarity = anchor_normalized * other_normalized
    # now multiply by the temperature.
    similarity *= args.nce_presence_temperature
    # and then sum to get the dot product (cosign similarity).
    # this is [bs, bs]. the positive samples are on the diagonal.
    similarity = similarity.sum(2)

    # # NOTE: We change to use pytorch here. Should be the same as ours though ...
    # similarity = F.cosine_similarity(anchor, other)
    # print('YO122: ', similarity)

    # the diagonal has the positive similarity = log(exp(sim(x, y)))
    identity = torch.eye(batch_size).to(similarity.device)
    positive_similarity = (similarity * identity).sum(1)

    # we get the total similarity by taking the logsumexp of similarity.
    log_sum_total = torch.logsumexp(similarity, dim=1)
    diff = positive_similarity - log_sum_total
    nce = -diff.mean()

    total_similarity = similarity.sum(1)
    negative_similarity = (total_similarity - positive_similarity).mean().item() / (batch_size - 1)
    positive_similarity = positive_similarity.mean().item()

    loss = nce * args.nce_presence_lambda
    stats = {}
    stats.update({
        'pos_sim_presence': positive_similarity,
        'neg_sim_presence': negative_similarity,
        'nce_presence': nce.item()
    })
    return loss, stats


def get_probs_test_loss(model, images, device, epoch, args):
    batch_size, num_images = images.shape[:2]

    # Change view so that we can put everything through the model at once.
    images = images.view(batch_size * num_images, *images.shape[2:])
    probs = model(images)

    probs_sum1 = probs.sum(dim=1, keepdim=True) + 1e-8
    within_presence = probs / probs_sum1 + 1e-8
    within_entropy = -within_presence * torch.log(within_presence) / np.log(2)
    within_entropy = within_entropy.sum(1)
    within_entropy = within_entropy.mean()

    probs_sum0 = probs.sum(dim=0) + 1e-8
    probs_sum0_distr = probs_sum0 / probs_sum0.sum(0)
    between_entropy = -probs_sum0_distr * torch.log(probs_sum0_distr) / np.log(2)
    between_entropy = between_entropy.mean(0)

    stats = {}
    stats['within_entropy'] = within_entropy.item()
    stats['between_entropy'] = between_entropy.item()
    max_values, _ = torch.max(probs, 1)
    min_values, _ = torch.min(probs, 1)
    stats['mean_max_prob'] = max_values.mean().item()
    stats['mean_min_prob'] = min_values.mean().item()
    stats['mean_prob'] = probs.mean().item()

    probs = probs.view(batch_size, num_images, *probs.shape[1:])

    loss = 0

    if args.use_simclr_nce:
        loss_nce, stats_nce = _do_simclr_nce(args.nce_presence_temperature, probs)
    else:
        loss_nce, stats_nce = _do_our_nce(probs, args)

    loss += loss_nce
    stats.update(stats_nce)
    return loss, stats


def get_backbone_test_loss(model, images, labels, device, epoch, args):
    # Images are expected to come as singular, not as [bs, num_images].
    output = model(images)
    loss = F.cross_entropy(output, labels)
    predictions = torch.argmax(output, dim=1)
    accuracy = (predictions == labels).float().mean().item()
    stats = {
        'accuracy': accuracy,
    }
    return loss, stats


def get_nceprobs_selective_loss(model, images, device, epoch, args,
                                num_class_capsules=10, store_dir=None):
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
    if store_dir:
        path = os.path.join(store_dir, 'images%d-%d.%03f.png')
        if epoch == 0 and args.num_routing == 1 and args.nceprobs_selection_temperature == 1:
            for num_set in range(min(15, images.shape[0])):
                for num_img in range(images.shape[1]):
                    img = images[num_set, num_img].cpu().numpy()
                    img = (img * 255).astype(np.uint8).squeeze()
                    imgpil = Image.fromarray(img)
                    path_ = path % (num_set, num_img, args.step_length)
                    imgpil.save(path_)

    batch_size, num_images = images.shape[:2]
    images = images.view(batch_size * num_images, *images.shape[2:])
    pose, presence_probs = model(images)
    pose = pose.view(batch_size, num_images, *pose.shape[1:])
    presence_probs_image = presence_probs.view(
        batch_size, num_images, -1)

    stats = {}
    loss = 0.

    # Get the loss for the nce over probs.
    nce, stats_ = _do_simclr_nce(args.nce_presence_temperature,
                                 presence_probs_image,
                                 selection_strategy=args.simclr_selection_strategy,
                                 do_norm=args.simclr_do_norm)
    loss += nce * args.nce_presence_lambda
    stats.update(stats_)

    presence_probs_image_detached = presence_probs_image.detach()
    if args.nceprobs_selection == 'ncelinear_maxfirst':
        # Note We don't know if the maximums are the same across the images.
        # The expectation is that they *will* be, but that's not obviously true.
        # So we could do this over *just* the first frame or we could pick a
        # random frame or we could multiply the probs together. Here we pick
        # just the first frame in order to make this predictable. Later we can
        # try other prediction mechanisms.
        max_indices = torch.argmax(presence_probs_image_detached[:, 0], dim=1)
        # shape of pose is [bs, nimg, ncaps, ndim], shape of indices is [bs].
        # Now we want to get the nth index from max_indices from the nth batch
        # entry and then combine them again.
        selected_capsules = torch.stack(
            [pose[num, :, index] for num, index in enumerate(max_indices)]
        )
        max_unique, max_counts = torch.unique(max_indices, return_counts=True)
        max_counts = max_counts.float()
        max_unique_counts, _ = torch.max(max_counts, 0)
        mean_unique_counts = torch.mean(max_counts)
        stats.update({
            'max_count_maximally_activate_capsule': max_unique_counts.item(),
            'mean_count_maximally_activate_capsule': mean_unique_counts.item(),
        })
        for index in max_indices:
            key = 'activated_%d' % index
            if key not in stats:
                stats[key] = 0
            stats[key] += 1

        # Shape of selected_capsules is now [bs, ni, 1, capsule_dim]. We seek:
        # nce(cossim(f_2 - f_0, f_1 - f_0) - cossim(f_2 - f_0, f'_1 - f'_0)).
        # We can get this by feeding _do_simclr_nce with those two vectors:
        if args.ncelinear_anchorselect == '21':
            anchor = selected_capsules[:, 2] - selected_capsules[:, 1]
        elif args.ncelinear_anchorselect == '20':
            anchor = selected_capsules[:, 2] - selected_capsules[:, 0]
        other = selected_capsules[:, 1] - selected_capsules[:, 0]
        nce_pose, stats_ = _do_simclr_nce(args.nceprobs_selection_temperature,
                                          anchor=anchor, other=other,
                                          suffix='selection',
                                          do_norm=args.simclr_do_norm)
        loss += nce_pose * args.nce_selection_lambda
        stats.update(stats_)

        # Get the capsule distance stats.
        sim_12 = torch.dist(selected_capsules[:, 0, :], selected_capsules[:, 1, :])
        sim_23 = torch.dist(selected_capsules[:, 1, :], selected_capsules[:, 2, :])
        sim_13 = torch.dist(selected_capsules[:, 0, :], selected_capsules[:, 2, :])
        cosine_sim_13_12 = F.cosine_similarity(anchor, other).mean()
        stats.update({
            'capsule_12_sim': sim_12.item(),
            'capsule_23_sim': sim_23.item(),
            'capsule_13_sim': sim_13.item(),
            'capsule_triangle_margin': (sim_13 - sim_12 - sim_23).item(),
            'capsule_cosine_sim_13_12': cosine_sim_13_12.item()
        })
    elif args.nceprobs_selection == 'ncelinear_threshfirst':
        pass
    elif args.nceprobs_selection == 'ncelinear_none':
        pass
    else:
        raise

    # Get the probability stats
    if 'sigmoid' in args.presence_loss_type:
        presence_probs_first = presence_probs_image[:, 0]
        presence_probs_sum1 = presence_probs_first.sum(dim=1, keepdim=True) + 1e-8
        within_presence = presence_probs_first / presence_probs_sum1 + 1e-8
        within_entropy = -within_presence * torch.log(within_presence) / np.log(2)
        within_entropy = within_entropy.sum(1)
        within_entropy = within_entropy.mean()

        presence_probs_sum0 = presence_probs_first.sum(dim=0) + 1e-8
        presence_probs_sum0_distr = presence_probs_sum0 / presence_probs_sum0.sum(0)
        between_entropy = -presence_probs_sum0_distr * torch.log(presence_probs_sum0_distr) / np.log(2)
        between_entropy = between_entropy.mean(0)

        max_values, _ = torch.max(presence_probs_first, 1)
        min_values, _ = torch.min(presence_probs_first, 1)
        mean_per_capsule = [value.item() for value in presence_probs.mean(0)]
        std_per_capsule = [value.item() for value in presence_probs.std(0)]
        l2_probs_12 = torch.norm(
            presence_probs_image[:, 0] - presence_probs_image[:, 1], dim=1).mean()
        l2_probs_13 = torch.norm(
            presence_probs_image[:, 0] - presence_probs_image[:, 2], dim=1).mean()
        l2_probs_23 = torch.norm(
            presence_probs_image[:, 1] - presence_probs_image[:, 2], dim=1).mean()

        for num, item in enumerate(mean_per_capsule):
            stats['capsule_prob_mean_%d' % num] = item
        for num, item in enumerate(std_per_capsule):
            stats['capsule_prob_std_%d' % num] = item

        stats.update({
            'within_entropy': within_entropy.item(),
            'between_entropy': between_entropy.item(),
            'mean_max_prob': max_values.mean().item(),
            'mean_min_prob': min_values.mean().item(),
            'mean_prob': presence_probs_first.mean().item(),
            'l2_presence_probs_12': l2_probs_12.item(),
            'l2_presence_probs_13': l2_probs_13.item(),
            'l2_presence_probs_23': l2_probs_23.item(),
        })
    elif 'l2norm' in args.presence_loss_type:
        # Here, we don't have values between 0 and 1. Instead, they are norms.
        presence_probs_first = presence_probs_image[:, 0]
        max_values, _ = torch.max(presence_probs_first, 1)
        min_values, _ = torch.min(presence_probs_first, 1)
        mean_per_capsule = [value.item() for value in presence_probs.mean(0)]
        std_per_capsule = [value.item() for value in presence_probs.std(0)]
        l2_probs_12 = torch.norm(
            presence_probs_image[:, 0] - presence_probs_image[:, 1], dim=1).mean()
        l2_probs_13 = torch.norm(
            presence_probs_image[:, 0] - presence_probs_image[:, 2], dim=1).mean()
        l2_probs_23 = torch.norm(
            presence_probs_image[:, 1] - presence_probs_image[:, 2], dim=1).mean()

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

def get_nceprobs_selective_reorder_loss(model, images, labels, args):
    # Change view so that we can put everything through the model at once.
    batch_size, num_images = images.shape[:2]
    images = images.view(batch_size * num_images, *images.shape[2:])
    pose, presence_probs, ordering, stats = model(images, return_ordering_selection=args.nceprobs_selection)
    # pose = pose.view(batch_size, num_images, *pose.shape[1:])
    presence_probs_image = presence_probs.view(
        batch_size, num_images, -1)

    # stats = {}
    loss = 0.

    # Get the loss for the nce over probs.
    nce, stats_ = _do_simclr_nce(args.nce_presence_temperature,
                                 presence_probs_image)
    loss += nce * args.nce_presence_lambda
    stats.update(stats_)

    # presence_probs_image_detached = presence_probs_image.detach()
    if args.nceprobs_selection == 'ncelinear_maxfirst':
        labels = labels.squeeze(-1).float().to(ordering.device)
        ordering_loss = F.binary_cross_entropy_with_logits(ordering, labels)
        loss += ordering_loss * args.lambda_ordering

        predictions = torch.sigmoid(ordering) > 0.5
        accuracy = (predictions == labels).float().mean().item()

        stats.update({
            'accuracy': accuracy,
            'ordering_loss': ordering_loss.item()
        })

    elif args.nceprobs_selection == 'ncelinear_threshfirst':
        pass
    else:
        raise

    # Get the probability stats
    presence_probs_first = presence_probs_image[:, 0]
    presence_probs_sum1 = presence_probs_first.sum(dim=1, keepdim=True) + 1e-8
    within_presence = presence_probs_first / presence_probs_sum1 + 1e-8
    within_entropy = -within_presence * torch.log(within_presence) / np.log(2)
    within_entropy = within_entropy.sum(1)
    within_entropy = within_entropy.mean()

    presence_probs_sum0 = presence_probs_first.sum(dim=0) + 1e-8
    presence_probs_sum0_distr = presence_probs_sum0 / presence_probs_sum0.sum(0)
    between_entropy = -presence_probs_sum0_distr * torch.log(presence_probs_sum0_distr) / np.log(2)
    between_entropy = between_entropy.mean(0)

    max_values, _ = torch.max(presence_probs_first, 1)
    min_values, _ = torch.min(presence_probs_first, 1)
    mean_per_capsule = [value.item() for value in presence_probs.mean(0)]
    std_per_capsule = [value.item() for value in presence_probs.std(0)]
    l2_probs_12 = torch.norm(
        presence_probs_image[:, 0] - presence_probs_image[:, 1], dim=1).mean()
    l2_probs_13 = torch.norm(
        presence_probs_image[:, 0] - presence_probs_image[:, 2], dim=1).mean()
    l2_probs_23 = torch.norm(
        presence_probs_image[:, 1] - presence_probs_image[:, 2], dim=1).mean()

    for num, item in enumerate(mean_per_capsule):
        stats['capsule_prob_mean_%d' % num] = item
    for num, item in enumerate(std_per_capsule):
        stats['capsule_prob_std_%d' % num] = item

    stats.update({
        'within_entropy': within_entropy.item(),
        'between_entropy': between_entropy.item(),
        'mean_max_prob': max_values.mean().item(),
        'mean_min_prob': min_values.mean().item(),
        'mean_prob': presence_probs_first.mean().item(),
        'l2_presence_probs_12': l2_probs_12.item(),
        'l2_presence_probs_13': l2_probs_13.item(),
        'l2_presence_probs_23': l2_probs_23.item(),
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

def old_get_nceprobs_selective_reorder_loss(model, images, labels, args):
    # Change view so that we can put everything through the model at once.
    batch_size, num_images = images.shape[:2]
    images = images.view(batch_size * num_images, *images.shape[2:])
    pose, presence_probs = model(images, return_ordering_selection=args.nceprobs_selection)
    pose = pose.view(batch_size, num_images, *pose.shape[1:])
    presence_probs_image = presence_probs.view(
        batch_size, num_images, -1)

    stats = {}
    loss = 0.

    # Get the loss for the nce over probs.
    nce, stats_ = _do_simclr_nce(args.nce_presence_temperature,
                                 presence_probs_image)
    loss += nce * args.nce_presence_lambda
    stats.update(stats_)

    presence_probs_image_detached = presence_probs_image.detach()
    if args.nceprobs_selection == 'ncelinear_maxfirst':
        max_indices = torch.argmax(presence_probs_image_detached[:, 0], dim=1)
        # shape of pose is [bs, nimg, ncaps, ndim], shape of indices is [bs].
        # Now we want to get the nth index from max_indices from the nth batch
        # entry and then combine them again.
        selected_capsules = torch.stack(
            [pose[num, :, index] for num, index in enumerate(max_indices)]
        )
        print(selected_capsules.shape)
        max_unique, max_counts = torch.unique(max_indices, return_counts=True)
        max_counts = max_counts.float()
        max_unique_counts, _ = torch.max(max_counts, 0)
        mean_unique_counts = torch.mean(max_counts)
        stats.update({
            'max_count_maximally_activate_capsule': max_unique_counts.item(),
            'mean_count_maximally_activate_capsule': mean_unique_counts.item(),
        })

        ordering = model(selected_capsules.view(batch_size, -1), return_ordering=True).squeeze(-1)
        labels = labels.squeeze(-1).float().to(ordering.device)
        ordering_loss = F.binary_cross_entropy_with_logits(ordering, labels)
        loss += ordering_loss * args.lambda_ordering

        predictions = torch.sigmoid(ordering) > 0.5
        accuracy = (predictions == labels).float().mean().item()

        stats.update({
            'accuracy': accuracy,
            'ordering_loss': ordering_loss.item()
        })

        # Get the capsule distance stats.
        sim_12 = torch.dist(selected_capsules[:, 0, :], selected_capsules[:, 1, :])
        sim_23 = torch.dist(selected_capsules[:, 1, :], selected_capsules[:, 2, :])
        sim_13 = torch.dist(selected_capsules[:, 0, :], selected_capsules[:, 2, :])
        stats.update({
            'capsule_12_sim': sim_12.item(),
            'capsule_23_sim': sim_23.item(),
            'capsule_13_sim': sim_13.item(),
        })
    elif args.nceprobs_selection == 'ncelinear_threshfirst':
        pass
    else:
        raise

    # Get the probability stats
    presence_probs_first = presence_probs_image[:, 0]
    presence_probs_sum1 = presence_probs_first.sum(dim=1, keepdim=True) + 1e-8
    within_presence = presence_probs_first / presence_probs_sum1 + 1e-8
    within_entropy = -within_presence * torch.log(within_presence) / np.log(2)
    within_entropy = within_entropy.sum(1)
    within_entropy = within_entropy.mean()

    presence_probs_sum0 = presence_probs_first.sum(dim=0) + 1e-8
    presence_probs_sum0_distr = presence_probs_sum0 / presence_probs_sum0.sum(0)
    between_entropy = -presence_probs_sum0_distr * torch.log(presence_probs_sum0_distr) / np.log(2)
    between_entropy = between_entropy.mean(0)

    max_values, _ = torch.max(presence_probs_first, 1)
    min_values, _ = torch.min(presence_probs_first, 1)
    mean_per_capsule = [value.item() for value in presence_probs.mean(0)]
    std_per_capsule = [value.item() for value in presence_probs.std(0)]
    l2_probs_12 = torch.norm(
        presence_probs_image[:, 0] - presence_probs_image[:, 1], dim=1).mean()
    l2_probs_13 = torch.norm(
        presence_probs_image[:, 0] - presence_probs_image[:, 2], dim=1).mean()
    l2_probs_23 = torch.norm(
        presence_probs_image[:, 1] - presence_probs_image[:, 2], dim=1).mean()

    for num, item in enumerate(mean_per_capsule):
        stats['capsule_prob_mean_%d' % num] = item
    for num, item in enumerate(std_per_capsule):
        stats['capsule_prob_std_%d' % num] = item

    stats.update({
        'within_entropy': within_entropy.item(),
        'between_entropy': between_entropy.item(),
        'mean_max_prob': max_values.mean().item(),
        'mean_min_prob': min_values.mean().item(),
        'mean_prob': presence_probs_first.mean().item(),
        'l2_presence_probs_12': l2_probs_12.item(),
        'l2_presence_probs_13': l2_probs_13.item(),
        'l2_presence_probs_23': l2_probs_23.item(),
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
