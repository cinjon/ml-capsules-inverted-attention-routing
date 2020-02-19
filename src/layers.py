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


#### Simple Backbone ####
class simple_backbone(nn.Module):

    def __init__(self, cl_input_channels, cl_num_filters, cl_filter_size,
                 cl_stride, cl_padding):
        super(simple_backbone, self).__init__()
        self.pre_caps = nn.Sequential(
            nn.Conv2d(in_channels=cl_input_channels,
                      out_channels=cl_num_filters,
                      kernel_size=cl_filter_size,
                      stride=cl_stride,
                      padding=cl_padding),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.pre_caps(x)  # x is an image
        return out


#### ResNet Backbone ####
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet_backbone(nn.Module):

    def __init__(self, cl_input_channels, cl_num_filters, cl_stride):
        super(resnet_backbone, self).__init__()
        self.in_planes = 64

        def _make_layer(block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        self.pre_caps = nn.Sequential(
            nn.Conv2d(in_channels=cl_input_channels,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            _make_layer(block=BasicBlock, planes=64, num_blocks=3,
                        stride=1),  # num_blocks=2 or 3
            _make_layer(block=BasicBlock,
                        planes=cl_num_filters,
                        num_blocks=4,
                        stride=cl_stride),  # num_blocks=2 or 4
        )

    def forward(self, x):
        out = self.pre_caps(x)  # x is an image
        return out


#### Capsule Layer ####
class CapsuleFC(nn.Module):
    r"""Applies as a capsule fully-connected layer.
    TBD
    """
    _name = 'CapsuleFC'

    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules,
                 out_d_capsules, matrix_pose, dp):
        super(CapsuleFC, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.matrix_pose = matrix_pose

        if matrix_pose:
            self.sqrt_d = int(np.sqrt(self.in_d_capsules))
            self.weight_init_const = np.sqrt(out_n_capsules /
                                             (self.sqrt_d * in_n_capsules))
            self.w = nn.Parameter(self.weight_init_const* \
                                          torch.randn(in_n_capsules, self.sqrt_d, self.sqrt_d, out_n_capsules))

        else:
            self.weight_init_const = np.sqrt(out_n_capsules /
                                             (in_d_capsules * in_n_capsules))
            self.w = nn.Parameter(self.weight_init_const* \
                                          torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules))
        self.dropout_rate = dp
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules**0.5)

    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, matrix_pose={}, \
            weight_init_const={}, dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules,
            self.out_d_capsules, self.matrix_pose, self.weight_init_const,
            self.dropout_rate)

    def forward(self, input, num_iter, next_capsule_value=None):
        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # m: num of capsules in next layer
        # d: dim of capsules in next layer
        if len(input.shape) == 5:
            input = input.permute(0, 4, 1, 2, 3)
            input = input.contiguous().view(input.shape[0], input.shape[1], -1)
            input = input.permute(0, 2, 1)

        if self.matrix_pose:
            w = self.w  # nxdm
            # This is [16,576,64] with a sqrt_d = 8 on the first go around.
            _input = input.view(input.shape[0], input.shape[1], self.sqrt_d, self.sqrt_d)
        else:
            w = self.w

        if next_capsule_value is None:
            query_key = torch.zeros(self.in_n_capsules,
                                    self.out_n_capsules).type_as(input)
            query_key = F.softmax(query_key, dim=1)

            if self.matrix_pose:
                next_capsule_value = torch.einsum('nm, bnax, nxdm->bmad',
                                                  query_key, _input, w)
            else:
                next_capsule_value = torch.einsum('nm, bna, namd->bmd',
                                                  query_key, input, w)
        else:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(
                    next_capsule_value.shape[0], next_capsule_value.shape[1],
                    self.sqrt_d, self.sqrt_d)
                _query_key = torch.einsum('bnax, nxdm, bmad->bnm', _input, w,
                                          next_capsule_value)
            else:
                _query_key = torch.einsum('bna, namd, bmd->bnm', input, w,
                                          next_capsule_value)
            _query_key.mul_(self.scale)
            query_key = F.softmax(_query_key, dim=2)
            query_key = query_key / (torch.sum(query_key, dim=2, keepdim=True) +
                                     1e-10)

            if self.matrix_pose:
                next_capsule_value = torch.einsum('bnm, bnax, nxdm->bmad',
                                                  query_key, _input, w)
            else:
                next_capsule_value = torch.einsum('bnm, bna, namd->bmd',
                                                  query_key, input, w)

        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            if self.matrix_pose:
                # [16, 10, 8, 8] ... with a self.out_d_capsules of 8
                next_capsule_value = next_capsule_value.view(
                    next_capsule_value.shape[0], next_capsule_value.shape[1],
                    self.out_d_capsules)
                next_capsule_value = self.nonlinear_act(next_capsule_value)
            else:
                next_capsule_value = self.nonlinear_act(next_capsule_value)
        return next_capsule_value


class CapsuleCONV(nn.Module):
    r"""Applies as a capsule convolutional layer.
    TBD
    """
    _name = 'CapsuleCONV'

    def __init__(self,
                 in_n_capsules,
                 in_d_capsules,
                 out_n_capsules,
                 out_d_capsules,
                 kernel_size,
                 stride,
                 matrix_pose,
                 dp,
                 coordinate_add=False):
        super(CapsuleCONV, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.kernel_size = kernel_size
        self.stride = stride
        self.matrix_pose = matrix_pose
        self.coordinate_add = coordinate_add

        if matrix_pose:
            self.sqrt_d = int(np.sqrt(self.in_d_capsules))
            self.weight_init_const = np.sqrt(
                out_n_capsules /
                (self.sqrt_d * in_n_capsules * kernel_size * kernel_size))
            self.w = nn.Parameter(
                self.weight_init_const *
                torch.randn(kernel_size, kernel_size, in_n_capsules,
                            self.sqrt_d, self.sqrt_d, out_n_capsules))
        else:
            self.weight_init_const = np.sqrt(
                out_n_capsules /
                (in_d_capsules * in_n_capsules * kernel_size * kernel_size))
            self.w = nn.Parameter(
                self.weight_init_const *
                torch.randn(kernel_size, kernel_size, in_n_capsules,
                            in_d_capsules, out_n_capsules, out_d_capsules))
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.dropout_rate = dp
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules**0.5)

    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, \
                    kernel_size={}, stride={}, coordinate_add={}, matrix_pose={}, weight_init_const={}, \
                    dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules,
            self.out_d_capsules, self.kernel_size, self.stride,
            self.coordinate_add, self.matrix_pose, self.weight_init_const,
            self.dropout_rate)

    def input_expansion(self, input):
        # input has size [batch x num_of_capsule x height x width x  x capsule_dimension]
        b, n, h, w, d = input.shape

        h_out = int((h - self.kernel_size) / self.stride + 1)
        w_out = int((w - self.kernel_size) / self.stride + 1)

        # this may be slow if the image size is large
        # TODO: kind of stupid implementation :'(
        inputs = torch.stack(
            [
                input[:, :, self.stride * i:self.stride * i +
                      self.kernel_size, self.stride * j:self.stride * j +
                      self.kernel_size, :]
                for i in range(h_out)
                for j in range(w_out)
            ],
            dim=-1)  # b,n, kernel_size, kernel_size, h_out*w_out*d
        inputs = inputs.view(b, n, self.kernel_size, self.kernel_size, h_out,
                             w_out, d)
        return inputs

    def forward(self, input, num_iter, next_capsule_value=None):
        # k,l: kernel size
        # h,w: output width and length
        inputs = self.input_expansion(input)

        if self.matrix_pose:
            w = self.w  # klnxdm
            _inputs = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3],\
                                  inputs.shape[4], inputs.shape[5], self.sqrt_d, self.sqrt_d) # bnklmhax
        else:
            w = self.w

        if next_capsule_value is None:
            query_key = torch.zeros(self.in_n_capsules, self.kernel_size,
                                    self.kernel_size,
                                    self.out_n_capsules).type_as(inputs)
            query_key = F.softmax(query_key, dim=3)

            if self.matrix_pose:
                next_capsule_value = torch.einsum(
                    'nklm, bnklhwax, klnxdm->bmhwad', query_key, _inputs, w)
            else:
                next_capsule_value = torch.einsum(
                    'nklm, bnklhwa, klnamd->bmhwd', query_key, inputs, w)
        else:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0],\
                                         next_capsule_value.shape[1], next_capsule_value.shape[2],\
                                         next_capsule_value.shape[3], self.sqrt_d, self.sqrt_d)
                _query_key = torch.einsum('bnklhwax, klnxdm, bmhwad->bnklmhw',
                                          _inputs, w, next_capsule_value)
            else:
                _query_key = torch.einsum('bnklhwa, klnamd, bmhwd->bnklmhw',
                                          inputs, w, next_capsule_value)
            _query_key.mul_(self.scale)
            query_key = F.softmax(_query_key, dim=4)
            query_key = query_key / (torch.sum(query_key, dim=4, keepdim=True) +
                                     1e-10)

            if self.matrix_pose:
                next_capsule_value = torch.einsum(
                    'bnklmhw, bnklhwax, klnxdm->bmhwad', query_key, _inputs, w)
            else:
                next_capsule_value = torch.einsum(
                    'bnklmhw, bnklhwa, klnamd->bmhwd', query_key, inputs, w)

        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(
                    next_capsule_value.shape[0],
                    next_capsule_value.shape[1],
                    next_capsule_value.shape[2],
                    next_capsule_value.shape[3],
                    self.out_d_capsules)
                next_capsule_value = self.nonlinear_act(next_capsule_value)
            else:
                next_capsule_value = self.nonlinear_act(next_capsule_value)

        return next_capsule_value


class CapsuleClassFC(CapsuleFC):
    """Class to represent the CapsuleClassFC.

    We include this to enable logging differentiated from CapsuleFC.
    """
    _name = 'CapsuleClassFC'
