# Object on the right is original object and object on the left is reconstruction
# Example command to visualize all data points:
# python pc_visualize.py --data_root /.../datasetFull --store_dir /.../vis_imgs --model_path /.../ckpt.epoch12.best.pth --mode full
# Example command to visualize a single data point:
# python pc_visualize.py --data_root /.../datasetFull --store_dir /.../vis_imgs --model_path /.../ckpt.epoch12.best.pth --mode single --data_num 1

import os
import argparse
import numpy as np
import open3d as o3d
from collections import OrderedDict

import torch

from src.shapenet import ShapeNet55
from src.capsule_ae import PointCapsNet

def o3d_visualize(original_points, reconstruction_points, store_dir, data_num):
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_points)
    reconstruction_pcd = o3d.geometry.PointCloud()
    reconstruction_pcd.points = o3d.utility.Vector3dVector(
        reconstruction_points + [2, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(original_pcd)
    vis.add_geometry(reconstruction_pcd)
    vis.poll_events()
    vis.capture_screen_image(os.path.join(store_dir, f'{data_num}.png'))
    vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--store_dir', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument(
        '--mode',
        default='single',
        type=str,
        help='single or full. If single then only visualize')
    parser.add_argument(
        '--data_num',
        default=None,
        type=int,
        help='the number of data point to visualize')
    parser.add_argument('--presence_type', default='l2norm', type=str)
    args = parser.parse_args()
    if args.mode == 'single':
        assert args.data_num is not None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading data')
    dataset = ShapeNet55(
        args.data_root, split='val', npoints=2048, num_frames=1,
        use_diff_object=False, stepsize_range=[0,0], stepsize_fixed=None,
        rotation_range=None, rotation_same=None)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, num_workers=2, shuffle=False)

    params = {
        'backbone': {
            'prim_caps_size': 1024,
            'prim_vec_size': 16,
            'num_points': 2048,
            'latent_caps_size': 64,
            'latent_vec_size': 64
        }
    }

    model = PointCapsNet(params, args)
    model = model.to(device)

    print('Loading model')
    state_dict = torch.load(args.model_path, map_location=device)['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    model.load_state_dict(new_state_dict)

    model.eval()

    os.makedirs(args.store_dir, exist_ok=True)

    with torch.no_grad():
        if args.mode == 'single':
            original_points, _ = dataset[args.data_num]
            original_points = original_points[0].unsqueeze(0).to(device)
            _, reconstruction_points = model(original_points)

            original_points = original_points.squeeze(0).permute(1, 0).cpu().numpy()
            reconstruction_points = reconstruction_points.squeeze(0).permute(1, 0).cpu().numpy()

            o3d_visualize(
                original_points, reconstruction_points, args.store_dir, args.data_num)

        elif args.mode == 'full':
            original_points_arr = []
            reconstruction_points_arr = []
            for batch_idx, (original_points, _) in enumerate(dataloader):
                if (batch_idx + 1) % 10 == 0:
                    print(f'Reconstruction {batch_idx+1}/{len(dataloader)}')

                original_points = original_points[:, 0].to(device)
                _, reconstruction_points = model(original_points)

                original_points_arr.append(
                    original_points.permute(0, 2, 1).cpu().numpy())
                reconstruction_points_arr.append(
                    reconstruction_points.permute(0, 2, 1).cpu().numpy())

            original_poitns_arr = np.asarray(
                original_points_arr).reshape((-1, 2048, 3))
            reconstruction_points_arr = np.asarray(
                reconstruction_points_arr).reshape((-1, 2048, 3))

            for i, (original_points, reconstruction_points) in\
                enumerate(zip(original_poitns_arr, reconstruction_points_arr)):

                if (i + 1) % 100 == 0:
                    print(f'Visualization {i+1}/{len(original_points)}')

                o3d_visualize(
                    original_points, reconstruction_points, args.store_dir, i+1)
