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
                 num_frames=4,
                 is_discriminating_model=False, # Use True if original model.
                 use_presence_probs=False,
                 presence_temperature=1.0,
                 presence_loss_type='sigmoid_l1'
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

        self.num_class_capsules = params['class_capsules']['num_caps']

        self.is_discriminating_model = is_discriminating_model
        if is_discriminating_model:
            ## After Capsule
            # fixed classifier for all class capsules
            self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
            # different classifier for different capsules
            #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))
        else:
            num_concatenated_dims = num_frames * params['class_capsules']['caps_dim'] * \
                params['class_capsules']['num_caps']
            # Either it's ordered correctly or not.
            self.ordering_head = nn.Linear(num_concatenated_dims, 1)

        self.get_mnist_head = mnist_classifier_head
        if mnist_classifier_head:
            num_params = params['class_capsules']['caps_dim'] * params['class_capsules']['num_caps']
            self.mnist_classifier_head = nn.Linear(num_params, 10)

        self.use_presence_probs = use_presence_probs
        self.presence_temperature = presence_temperature
        self.presence_loss_type = presence_loss_type
        if use_presence_probs:
            input_dim = params['backbone']['output_dim']
            input_dim *= params['backbone']['out_img_size']**2
            output_dim = params['class_capsules']['num_caps']
            self.presence_prob_head = nn.Linear(input_dim, output_dim)

    def get_presence(self, pre_caps):
        logits = self.presence_prob_head(pre_caps.view(pre_caps.shape[0], -1))
        if self.presence_loss_type in [
                'sigmoid_l1', 'sigmoid_prior_sparsity',
                'sigmoid_prior_sparsity_example', 'sigmoid_within_entropy',
                'sigmoid_within_between_entropy',
                'sigmoid_prior_sparsity_example_between_entropy'
        ]:
            logits *= self.presence_temperature
            # NOTE: we add noise here in order to try and spike it.
            rand_noise = torch.FloatTensor(logits.size()).uniform_(-2, 2).to(logits.device)
            logits += rand_noise
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
        return logits

    def get_ordering(self, x):
        return self.ordering_head(x)

    def forward(self, x, lbl_1=None, lbl_2=None, return_embedding=False,
                flatten=False, return_mnist_head=False):
        #### Forward Pass
        ## Backbone (before capsule)
        # NOTE: pre_caps is [36, 128, 32, 32], or
        # [batch_size*num_images, backone['output_dim'],
        #  backbone['output_image_size], backbone['output_image_size']]
        # Lulz, this is 32x the input shape for mnist of [36, 1, 64, 64].
        c = self.pre_caps(x)
        if self.use_presence_probs:
            presence_probs = self.get_presence(c)

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
        if not self.sequential_routing:
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

        # NOTE: This is a triple of next_capsule_value, presence, object_
        # Pose is the next_capsule_value. So we have pose, presence, object.
        # We then want the presence to be sparse over the capsules (dim=1), so
        # we put an L1 penalty on it. Otherwise, it will just be 1 everywhere.
        # We want pose * object_ * presence to be informative of the ordering.
        # And we want pose * object_ to be relatively the same across frames.
        # ordering.
        # pose, presence, object_ = out
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
            out = pose.view(pose_shape[0], -1)
            out = self.mnist_classifier_head(out)
            return out, pose
        elif self.use_presence_probs:
            return pose, presence_probs
        else:
            # return pose, presence, object_
            return pose


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
        pose *= presence_probs[:, :, None]
    else:
        pose = model(images)
    pose = pose.view(batch_size, num_images, *pose.shape[1:])

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

        within_presence_norm = presence_probs.norm(dim=1, keepdim=True) + 1e-6
        within_presence_normed = presence_probs / within_presence_norm + 1e-6
        within_entropy = -within_presence_normed * \
            torch.log(within_presence_normed) / np.log(2)
        within_entropy = within_entropy.sum(1)

        presence_probs_image = presence_probs.view(
            batch_size, num_images, -1)
        presence_probs_first = presence_probs_image[:, 0]
        between_presence_norm = presence_probs_first.norm(dim=0, keepdim=True) + 1e-6
        between_presence_normed = presence_probs_first / between_presence_norm + 1e-6
        between_entropy = -between_presence_normed * \
            torch.log(between_presence_normed) / np.log(2)
        between_entropy = between_entropy.sum(0)
        
        within_entropy = within_entropy.mean()
        between_entropy = between_entropy.mean()        
        stats['within_entropy'] = within_entropy.item()
        stats['between_entropy'] = between_entropy.item()

        if args.presence_loss_type == 'sigmoid_l1':
            # NOTE: Using this presence loss results in the model always using
            # the same subset of capsules, regardless of input.
            presence_loss = presence_probs.sum(1).mean()
            loss += presence_loss * args.lambda_sparse_presence
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
            loss -= between_entropy * args.lambda_sparse_presence
            stats['example_presence_loss'] = example_presence_loss.item()
            stats['presence_loss'] = presence_loss.item()
        elif args.presence_loss_type == 'sigmoid_within_entropy':
            # Here, we use entropy constraints. We normalize the probabilities
            # and we try to minimize the within-example entropy. Minimizing the
            # entropy would result in a sharp distribution.
            # I suspect that this would enable the model to use few of the
            # capsules and be fine. Let's see if that happens.
            # So what this does is the model uses few of the capsules but then
            # doesn't spread them around. Everyone just uses the capsules.
            loss += within_entropy * args.lambda_sparse_presence
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
            loss += within_entropy * args.lambda_sparse_presence
            loss -= between_entropy * args.lambda_sparse_presence
        elif args.presence_loss_type == 'softmax':
            # This seems to result in everyone getting the same capsule. Let's
            # see what happens if we let it go longer.
            pass
        elif args.presence_loss_type == 'softmax_nonoise':
            # This also results in everyone getting the same capsule.
            pass
        elif args.presence_loss_type == 'softmax_within_between_entropy':
            loss += within_entropy * args.lambda_sparse_presence
            loss -= between_entropy * args.lambda_sparse_presence
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
