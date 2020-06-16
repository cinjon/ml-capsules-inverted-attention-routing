#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
'''Capsule in PyTorch
TBD
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


#### ResNet Backbone ####
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes,
                               planes,
                               kernel_size=1, # 3,
                               stride=stride,
                               # padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes,
                               planes,
                               kernel_size=1, # 3,
                               stride=1,
                               # padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm1d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResnetBackbone(nn.Module):

    def __init__(self, cl_input_channels, cl_num_filters, cl_stride, args=None):
        super(ResnetBackbone, self).__init__()
        cinjon_is_dumb = args.config == 'resnet_backbone_points16_3conv1fc' 
        if cinjon_is_dumb:
            self.in_planes = 64
        else:
            self.in_planes = cl_num_filters

        def _make_layer(block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        out_channels = 64 if cinjon_is_dumb else cl_num_filters
        print('Backeone: ', args.config, cinjon_is_dumb, self.in_planes, cl_num_filters)
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels=cl_input_channels,
                      out_channels=out_channels, # was 64
                      kernel_size=1, # 3,
                      stride=1,
                      # padding=1,
                      bias=False),
            nn.BatchNorm1d(out_channels), # was 64
            nn.ReLU(),
            _make_layer(block=BasicBlock, planes=out_channels, num_blocks=3,
                        stride=1),  # num_blocks=2 or 3
            _make_layer(block=BasicBlock,
                        planes=cl_num_filters,
                        num_blocks=4,
                        stride=cl_stride),  # num_blocks=2 or 4
        )

    def forward(self, x):
        return self.backbone(x)


# class GeometricBackbone(nn.Module):

#     def __init__(self, cl_input_channels, cl_num_filters, cl_stride):
#         super(ResnetBackbone, self).__init__()

#         def _make_layer(block, planes, num_blocks, stride):
#             strides = [stride] + [1] * (num_blocks - 1)
#             layers = []
#             for stride in strides:
#                 layers.append(block(self.in_planes, planes, stride))
#                 self.in_planes = planes * block.expansion
#             return nn.Sequential(*layers)

#         self.backbone = nn.Sequential(
#             nn.Conv1d(in_channels=cl_input_channels,
#                       out_channels=cl_num_filters, # was 64
#                       kernel_size=1, # 3,
#                       stride=1,
#                       # padding=1,
#                       bias=False),
#             nn.BatchNorm1d(cl_num_filters), # was 64
#             nn.ReLU(),
#             _make_layer(block=BasicBlock, planes=cl_num_filters, num_blocks=3,
#                         stride=1),  # num_blocks=2 or 3
#             _make_layer(block=BasicBlock,
#                         planes=cl_num_filters,
#                         num_blocks=4,
#                         stride=cl_stride),  # num_blocks=2 or 4
#         )

#     def forward(self, x):
#         return self.backbone(x)


#### Capsule Layer ####
class CapsuleFC(nn.Module):
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules,
                 out_d_capsules, gap=False):
        super(CapsuleFC, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.use_gap = gap

        ###
        # NOTE: Assuming matrix pose here.
        self.sqrt_d = int(np.sqrt(self.in_d_capsules))
        self.weight_init_const = np.sqrt(out_n_capsules /
                                         (self.sqrt_d * in_n_capsules))
        w = torch.randn(in_n_capsules, self.sqrt_d, self.sqrt_d, out_n_capsules)
        w = self.weight_init_const * w
        self.w = nn.Parameter(w)
        ###
        self.layer_norm = nn.LayerNorm(out_d_capsules)
        self.scale = 1. / (out_d_capsules**0.5)

    def apply_nonlinearity(self, next_capsule_value):
        next_capsule_value = next_capsule_value.view(
            next_capsule_value.shape[0], next_capsule_value.shape[1],
            self.out_d_capsules)
        return self.layer_norm(next_capsule_value)

    @staticmethod
    def maybe_permute_input(input):
        # 32,16,7,7,36  OR  32,10,36
        if len(input.shape) == 4:
            input = input.permute(0, 3, 1, 2)
            input = input.contiguous().view(input.shape[0], input.shape[1], -1)
            input = input.permute(0, 2, 1)
        return input

    def forward(self, input, num_iter, next_capsule_value=None):
        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # m: num of capsules in next layer
        # d: dim of capsules in next layer

        # if input from conv: [bs, num_capsules_prev, "img_size", caps_dim]
        # if input from fc: [bs, num_capsules_prev, caps_dim]
        if self.use_gap and len(input.shape) == 4:
            input = torch.mean(input, dim=2)

        pose = self.maybe_permute_input(input)

        w = self.w
        # pose is [24,32,509,16], input is [24, 32, 509, 16], w is [800, 4, 4, 25]
        _pose = pose.view(pose.shape[0], pose.shape[1], self.sqrt_d,
                          self.sqrt_d)

        if next_capsule_value is None:
            query_key = torch.zeros(self.in_n_capsules,
                                    self.out_n_capsules).type_as(pose)
            query_key = F.softmax(query_key, dim=1)
            next_capsule_value = torch.einsum('nm, bnax, nxdm->bmad',
                                              query_key, _pose, w)
            # The query key now is [10, 10]
        else:
            # NOTE: This right here is a conflagration of the vote and and the
            # pose of the current capsule in order to get the query_key.
            # Is some function of the vote what we care about?
            next_capsule_value = next_capsule_value.view(
                next_capsule_value.shape[0], next_capsule_value.shape[1],
                self.sqrt_d, self.sqrt_d)
            _query_key = torch.einsum('bnax, nxdm, bmad->bnm', _pose, w,
                                      next_capsule_value)
            _query_key.mul_(self.scale)
            query_key = F.softmax(_query_key, dim=2)
            # 32,784,10
            query_key = query_key / (torch.sum(query_key, dim=2, keepdim=True) +
                                     1e-10)
            next_capsule_value = torch.einsum('bnm, bnax, nxdm->bmad',
                                              query_key, _pose, w)

        # 32,10,6,6
        if not next_capsule_value.shape[-1] == 1:
            next_capsule_value = self.apply_nonlinearity(next_capsule_value)

        pose = next_capsule_value
        return pose


class CapsuleConv(nn.Module):
    def __init__(self,
                 in_n_capsules,
                 in_d_capsules,
                 out_n_capsules,
                 out_d_capsules,
                 kernel_size,
                 stride):
        super(CapsuleConv, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.kernel_size = kernel_size
        self.stride = stride

        # NOTE: Assuming matrix pose here.
        self.sqrt_d = int(np.sqrt(self.in_d_capsules))
        self.weight_init_const = np.sqrt(
            out_n_capsules /
            (self.sqrt_d * in_n_capsules * kernel_size * kernel_size))
        self.w = nn.Parameter(
            self.weight_init_const *
            torch.randn(kernel_size, in_n_capsules, self.sqrt_d, self.sqrt_d,
                        out_n_capsules))

        self.layer_norm = nn.LayerNorm(out_d_capsules)
        self.scale = 1. / (out_d_capsules**0.5)

    def input_expansion(self, input):
        # TODO: Is this still ok if we are using 1d? I think so ...
        unfolded_input = input.unfold(2, size=self.kernel_size, step=self.stride)
        # When we do the unfold twice, once over dim 2 and then over dim 3:
        # input.shape: [bs, num_caps, img_size, num_cap_dims]
        # unfolded_input: [bs, num_caps, ~img_size/2, 7, 3, 3] (hrm. hard to know)
        # When we do it just over dim 2:
        # input.shape: [bs, num_caps, img_size, num_cap_dims]
        # unfolded_input: [bs, num_caps, ~img_size/2, 16, 3] (hrm. hard to know)
        unfolded_input = unfolded_input.permute([0, 1, 4, 2, 3])
        # output is [bs, num_caps, kernel, strided out (w no padding), caps_dim]
        return unfolded_input

    def forward(self, input, num_iter, next_capsule_value=None):
        # k,l: kernel size
        # h,w: output width and length
        inputs = self.input_expansion(input)
        # _inputs is shape [bs, num_caps, k, w, sqrtd, sqrtd]
        _inputs = inputs.view(
            inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3],\
            self.sqrt_d, self.sqrt_d) # bnklmhax
        w = self.w  # klnxdm

        if next_capsule_value is None:
            query_key = torch.zeros(self.in_n_capsules, self.kernel_size,
                                    self.out_n_capsules).type_as(inputs)
            query_key = F.softmax(query_key, dim=2)
            next_capsule_value = torch.einsum(
                'nkm, bnkwax, knxdm->bmwad', query_key, _inputs, w)
        else:
            next_capsule_value = next_capsule_value.view(
                next_capsule_value.shape[0], next_capsule_value.shape[1],
                next_capsule_value.shape[2], next_capsule_value.shape[3],
                self.sqrt_d, self.sqrt_d
            )
            _query_key = torch.einsum('bnkwax, knxdm, bmwad->bnkmw',
                                      _inputs, w, next_capsule_value)
            _query_key.mul_(self.scale)
            query_key = F.softmax(_query_key, dim=3)
            # TODO: the below shoul be changed to ... dim=3 as well?
            query_key = query_key / (torch.sum(query_key, dim=4, keepdim=True) +
                                     1e-10)
            next_capsule_value = torch.einsum(
                'bnkmw, bnkwax, knxdm->bmwad', query_key, _inputs, w)

        if not next_capsule_value.shape[-1] == 1:
            # torch.Size([24, 32, 509, 4, 4]) 16
            next_capsule_value = next_capsule_value.view(
                next_capsule_value.shape[0], next_capsule_value.shape[1],
                next_capsule_value.shape[2], self.out_d_capsules)
            # ncvvvvshapes:  torch.Size([24, 32, 509, 16])
            next_capsule_value = self.layer_norm(next_capsule_value)
            # ncvvvvshapes33:  torch.Size([24, 32, 509, 16])

        return next_capsule_value
