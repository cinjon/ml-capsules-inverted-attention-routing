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
                params['backbone']['output_dim'], params['backbone']['stride'])

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

        num_concatenated_dims = 3 * params['class_capsules']['caps_dim'] * \
            params['class_capsules']['num_caps']
        # Either it's ordered correctly or not.
        self.ordering_head = nn.Linear(num_concatenated_dims, 1)
        self.get_mnist_head = mnist_classifier_head
        if mnist_classifier_head:
            self.mnist_classifier_head = nn.Linear(params['class_capsules']['caps_dim'], 1)

    def forward(self, x, lbl_1=None, lbl_2=None, return_embedding=False, flatten=False):
        #### Forward Pass
        ## Backbone (before capsule)
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
                return out.view(out.size(0), -1)
            else:
                return out
        elif self.get_mnist_head:
            out = pose
            out = self.mnist_classifier_head(out).squeeze()
        else:
            # return pose, presence, object_
            return pose

    def get_bce_loss(self, images, labels):
        images = images[:, 0, :, :, :]
        output = self(images, return_embedding=False)
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

    def get_xent_loss(self, images, labels):
        output = self(images, return_embedding=False)
        loss = F.cross_entropy(output, labels)
        predictions = torch.argmax(output, dim=1)
        accuracy = (predictions == labels).float().mean().item()
        stats = {
            'accuracy': accuracy,            
        }
        return loss, stats

    def get_triplet_loss(self, images, args):
        inputs_v = self(inputs, return_embedding=True, flatten=False)
        positive_v = self(positive, return_embedding=True, flatten=False)
        # TODO: if batch size is odd num then it won't work
        negative_v = positive_v.flip(0)

        # Compute distance
        with torch.no_grad():
            positive_distance = float(torch.dist(inputs_v, positive_v, 2).item())
            negative_distance = float(torch.dist(inputs_v, negative_v, 2).item())

        stats = {'positive_distance': positive_distance,
                 'negative_distance': negative_distance}

        return loss, stats

    def get_nce_loss(self, images, args):
        positive_frame_num = args.nce_positive_frame_num
        use_random_anchor_frame = args.use_random_anchor_frame

        if use_random_anchor_frame:
            # NOTE: not implemented.
            raise
        else:
            anchor_frame = 0

        anchor = images[:, anchor_frame]
        anchor = self(anchor, return_embedding=True, flatten=False)
        other = images[:, anchor_frame + positive_frame_num]
        other = self(other, return_embedding=True, flatten=False)

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

    def get_reorder_loss(self, images, device, args, labels=None):
        # images come in as [bs, num_imgs, ch, w, h]. we want to pick from
        # this three frames to use as either positive or negative.

        # select frames (a, b, c, d, e)
        range_size = int(images.shape[1] / 5)
        # if labels is not None:
        #     path = '/misc/kcgscratch1/ChoGroup/resnick/vid%d.frame%d.after.png'
        #     for batch_num in range(len(images)):
        #         image = images[batch_num].numpy()
        #         label_ = labels[batch_num].numpy()
        #         for num in range(len(image)):
        #             arr = image[num]
        #             arr = (255 * arr).astype(np.uint8)
        #             img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
        #             img.save(path % (label_, num))

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

        # if labels is not None:
        #     path = '/misc/kcgscratch1/ChoGroup/resnick/vid%d.frame%d.after%d.png'
        #     for batch_num in range(len(images)):
        #         image = images[batch_num].numpy()
        #         label_ = labels[batch_num].numpy()
        #         for num in range(len(image)):
        #             arr = image[num]
        #             arr = (255 * arr).astype(np.uint8)
        #             img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
        #             img.save(path % (label_, num, int(use_positive)))

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
        pose = self(images)
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
        ordering = self.ordering_head(flattened).squeeze(-1)

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
        
