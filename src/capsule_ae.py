from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class PrimaryPointCapsLayer(nn.Module):
    def __init__(self, prim_caps_size=1024, prim_vec_size=8, num_points=2048):
        super(PrimaryPointCapsLayer, self).__init__()
        self.prim_caps_size = prim_caps_size
        self.prim_vec_size = prim_vec_size
        self.num_points = num_points

        self.capsules = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', nn.Conv1d(128, self.prim_caps_size, 1)),
                ('bn3', nn.BatchNorm1d(self.prim_caps_size)),
                ('mp1', nn.MaxPool1d(self.num_points)),
            ]))
            for _ in range(self.prim_vec_size)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=2)
        return self.squash(u.squeeze())

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        if(output_tensor.dim() == 2):
            output_tensor = torch.unsqueeze(output_tensor, 0)
        return output_tensor

class LatentCapsLayer(nn.Module):
    def __init__(self, latent_caps_size=16, prim_caps_size=1024, prim_vec_size=16, latent_vec_size=64):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_caps_size = latent_caps_size
        self.W = nn.Parameter(0.01*torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach()
        b_ij = torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size).to(x.device)
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)
        return v_j.squeeze(-2)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        super(PointGenCon, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size/4), 3, 1)
        self.th = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size/2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size/4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class CapsDecoder(nn.Module):
    def __init__(self, latent_caps_size, latent_vec_size, num_points):
        super(CapsDecoder, self).__init__()
        self.latent_caps_size = latent_caps_size
        self.bottleneck_size = latent_vec_size
        self.num_points = num_points
        self.nb_primitives = int(num_points/latent_caps_size)
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size+2) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = torch.FloatTensor(x.size(0), 2, self.latent_caps_size).to(x.device)
            rand_grid.data.uniform_(0, 1)
            y = torch.cat((rand_grid, x.transpose(2, 1)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()

class NewBackboneModel(nn.Module):
    def __init__(self, params, args, out_channels=None):
        super(NewBackboneModel, self).__init__()
        backbone = params['backbone']
        self.prim_caps_size = backbone['prim_caps_size']
        self.prim_vec_size = backbone['prim_vec_size']
        self.latent_caps_size = backbone['latent_caps_size'] # 64
        self.latent_vec_size = backbone['latent_vec_size'] # 64
        self.num_points = backbone['num_points']


        if out_channels is None:
            self.out_channels = args.num_output_classes
        else:
            self.out_channels = out_channels

        self.presence_type = args.presence_type
        self.is_classifier = 'xent' in args.criterion
        self.dynamic_routing = args.dynamic_routing

        self.conv_layer = ConvLayer()
        self.primary_point_caps_layer = PrimaryPointCapsLayer(
            self.prim_caps_size, self.prim_vec_size, self.num_points)

        if self.dynamic_routing:
            self.latent_caps_layer = LatentCapsLayer(
                self.latent_caps_size, self.prim_caps_size,
                self.prim_vec_size, self.latent_vec_size)

        if self.is_classifier:
            if self.dynamic_routing:
                self.fc_head = nn.Linear(
                    self.latent_caps_size * self.latent_vec_size, self.out_channels)
            else:
                self.fc_head = nn.Linear(
                    self.prim_caps_size * self.prim_vec_size, self.out_channels)

    def forward(self, x, return_embedding=False):
        x = self.conv_layer(x)
        x = self.primary_point_caps_layer(x)

        if return_embedding:
            return x

        # Dynamic routing
        if self.dynamic_routing:
            x = self.latent_caps_layer(x)


        presence = self.get_presence(x)

        if self.is_classifier:
            x = self.fc_head(x.view(x.size(0), -1))
        return x, presence

    def get_presence(self, final_pose):
        if not self.presence_type:
            return None
        elif self.presence_type == 'l2norm':
            return final_pose.norm(dim=2)

class PointCapsNet(nn.Module):
    def __init__(self, params, args):
        super(PointCapsNet, self).__init__()
        backbone = params['backbone']
        self.prim_caps_size = backbone['prim_caps_size']
        self.prim_vec_size = backbone['prim_vec_size']
        self.latent_caps_size = backbone['latent_caps_size']
        self.latent_vec_size = backbone['latent_vec_size']
        self.num_points = backbone['num_points']

        self.presence_type = args.presence_type

        self.conv_layer = ConvLayer()
        self.primary_point_caps_layer = PrimaryPointCapsLayer(
            self.prim_caps_size, self.prim_vec_size, self.num_points)
        self.latent_caps_layer = LatentCapsLayer(
            self.latent_caps_size, self.prim_caps_size,
            self.prim_vec_size, self.latent_vec_size)
        self.caps_decoder = CapsDecoder(
            self.latent_caps_size, self.latent_vec_size, self.num_points)

    def forward(self, x, get_code=False):
        x = self.conv_layer(x)

        x = self.primary_point_caps_layer(x)
        x = self.latent_caps_layer(x)

        if get_code:
            presence = self.get_presence(x)
            return x, presence
        else:
            reconstructions = self.caps_decoder(x)
            return x, reconstructions

    def get_presence(self, code):
        if not self.presence_type:
            return None
        elif self.presence_type == 'l2norm':
            return code.norm(dim=2)

    def loss(self, data, reconstructions):
        return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data = data.transpose(2, 1).contiguous()
        reconstructions = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2, _, _ = chamfer_dist(data, reconstructions)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss


def get_autoencoder_loss(model, images, args):
    import src.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
    chamfer_dist = dist_chamfer_3D.chamfer_3DDist()
    _, reconstructions = model(images)
    images = images.transpose(2, 1).contiguous()
    reconstructions = reconstructions.transpose(2, 1).contiguous()
    dist1, dist2, _, _ = chamfer_dist(images, reconstructions)

    dist1_mean = torch.mean(dist1)
    dist2_mean = torch.mean(dist2)
    loss = dist1_mean + dist2_mean
    stats = {
        'dist1_mean': dist1_mean.item(),
        'dist2_mean': dist2_mean.item()}
    return loss, stats


def get_xent_loss(model, points, labels):
    # Assumes that it's the NewBackboneModel
    output, _ = model(points)
    loss = F.cross_entropy(output, labels)
    predictions = torch.argmax(output, dim=1)
    accuracy = (predictions == labels).float().mean().item()
    stats = {
        'accuracy': accuracy,
    }
    return loss, stats
