import os
import argparse
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./ModelNet40')
parser.add_argument('--store_dir', type=str, default='./data')
args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(args.store_dir, exist_ok=True)

    folder_names = os.listdir(args.data_root)
    folder_paths = [os.path.join(args.data_root, folder_name) for folder_name in folder_names]
    splits = ['train', 'test']

    for i, (folder_name, folder_path) in enumerate(zip(folder_names, folder_paths)):
        for split in splits:
            os.makedirs(os.path.join(args.store_dir, folder_name, split), exist_ok=True)

            category_path = os.path.join(folder_path, split)
            mesh_names = os.listdir(category_path)
            mesh_paths = [os.path.join(category_path, mesh_name) for mesh_name in mesh_names]

            for j, (mesh_name, mesh_path) in enumerate(zip(mesh_names, mesh_paths)):
                store_path = os.path.join(args.store_dir, folder_name, split, f'{mesh_name[:-4]}.npy')
                if os.path.exists(store_path):
                    continue

                newf = ''
                corrupt = False
                with open(mesh_path, 'r') as f:
                    for line_i, line in enumerate(f):
                        if line_i == 0 and len(line) != 4:
                            newf += line[:3] + '\n' + line[3:]
                            corrupt = True
                        else:
                            newf += line

                        if not corrupt:
                            break

                if corrupt:
                    with open(mesh_path, 'w') as f:
                        f.write(newf)

                mesh = o3d.io.read_triangle_mesh(mesh_path)
                pcd = mesh.sample_points_uniformly(number_of_points=2048)
                pcd_array = np.asarray(pcd.points)
                np.save(args.store_path, pcd_array)
