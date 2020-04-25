import getpass
import os
import pickle
import random
import shutil

import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
import torchvision.transforms as transforms


class GymnasticsRgbFrame(data.Dataset):
    """The video directory below is a directory of directories of frames."""
    def __init__(self, transforms=None, train=True, skip_videoframes=5,
                 num_videoframes=100, dist_videoframes=50,
                 video_directory=None, video_names=None,
                 is_reorder_loss=False, is_triangle_loss=False,
                 positive_ratio=0.5, tau_min=15, tau_max=60):
        print('Copying the files over ...', video_names)
        user = getpass.getuser()
        path = '/scratch/%s/gymnastics/frames' % user
        if not os.path.exists(path):
            os.makedirs(path)
        for video_name in video_names:
            new_path = os.path.join(path, video_name + '.fps25')
            original_path = os.path.join(video_directory, video_name + '.fps25')
            if not os.path.exists(new_path):
                shutil.copytree(original_path, new_path)
        video_directory = path
        print('Done copying the files over.')

        self.train = train
        self.transforms = transforms
        self.video_directory = video_directory
        self.skip_videoframes = skip_videoframes
        self.num_videoframes = num_videoframes
        self.dist_videoframes = dist_videoframes
        self.is_reorder_loss = is_reorder_loss
        self.is_triangle_loss = is_triangle_loss
        self.is_normal_video = not is_reorder_loss and not is_triangle_loss
        self.positive_ratio = positive_ratio
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.frame_directories = sorted([
            os.path.join(video_directory, f) for f in os.listdir(video_directory) \
        ])
        if video_names:
            self.frame_directories = [
                v for v in self.frame_directories \
                if any([name in v for name in video_names]) \
                and not v.endswith('pkl')
            ]

        print("Frame directories: ", self.frame_directories, video_names)

        clip_length_in_frames = self.num_videoframes * self.skip_videoframes
        frames_between_clips = self.dist_videoframes
        midfix = '-' + '.'.join(video_names) + '-' if video_names else ''
        self.saved_frame_info = os.path.join(
            video_directory, 'frame_info%s%df.%ds.pkl' % (
                midfix, clip_length_in_frames, frames_between_clips))
        if os.path.exists(self.saved_frame_info):
            print('Path Exists for frame_info: ', self.saved_frame_info)
            self.frame_info = pickle.load(open(self.saved_frame_info, 'rb'))
        else:
            print('Path does NOT exist for frame_info: ', self.saved_frame_info)
            self.frame_info = self._get_frame_info(
                self.frame_directories, clip_length_in_frames=clip_length_in_frames,
                frames_between_clips=frames_between_clips, skip_frames=skip_videoframes)
            pickle.dump(self.frame_info, open(self.saved_frame_info, 'wb'))

        self.datums = self.frame_info
        print('Size of datums: ', len(self.datums))

    def __len__(self):
        return len(self.datums)

    def _get_frame_info(self, frame_directories, clip_length_in_frames,
                        frames_between_clips, skip_frames):
        # The videos were originally taken at 25 fps.
        ret = []
        for directory in frame_directories:
            total_frames = len([k for k in os.listdir(directory) \
                                if k.endswith('npy')])
            frame_index = 0
            clip_index = 0
            while frame_index < total_frames:
                end_index = frame_index + clip_length_in_frames
                if end_index < total_frames:
                    ret.append((directory, frame_index, clip_index))

                frame_index = end_index + frames_between_clips
                clip_index += 1
        return ret

    def __getitem__(self, index):
        directory, frame_index, clip_index = self.datums[index]
        frames = [frame_index + i*self.skip_videoframes
                  for i in range(self.num_videoframes)]
        frames = ['{0:.4f}.npy'.format(f / 25) for f in frames]
        frames = [os.path.join(directory, f) for f in frames]
        frames = [np.load(f) for f in frames]
        # path = '/misc/kcgscratch1/ChoGroup/resnick/vid%d.frame%d.before.png'
        # for num, frame in enumerate(frames):
        #     arr = frames[num]
        #     arr = (255 * arr).astype(np.uint8)
        #     img = Image.fromarray(np.transpose(arr, (1, 0, 2)))
        #     img.save(path % (index, num))

        if self.is_normal_video:
            video = torch.stack([self.transforms(frame) for frame in frames])
            return video, index

        range_size = int(self.num_videoframes / 5)
        sample = [random.choice(range(i*range_size, (i+1)*range_size))
                  for i in range(5)]

        # Maybe flip the list's order.
        if random.random() > 0.5:
            sample = list(reversed(sample))

        if self.is_triangle_loss:
            use_positive = True
            p = random.random()
            if p > 0.66:
                selection = sample[:3]
            elif p > 0.33:
                selection = sample[1:4]
            else:
                selection = sample[2:]
        else:
            images = [frames[sample[i]] for i in range(len(sample))]
            images = torch.stack([self.transforms(img) for img in images])
            # Compare ssd to tau
            ssd_a_b = get_ssd(images[0], images[1])
            ssd_d_e = get_ssd(images[3], images[4])
            ssd_b_d = get_ssd(images[1], images[3])

            if not (min(ssd_a_b, ssd_d_e) > self.tau_min and ssd_b_d < self.tau_max):
                return None

            use_positive = random.random() < self.positive_ratio
            if use_positive:
                video = images[1:4]
            elif random.random() > 0.5:
                video = torch.stack([images[1], images[0], images[3]])
            else:
                video = torch.stack([images[1], images[4], images[3]])


        #     use_positive = random.random() < self.positive_ratio
        #     if use_positive:
        #         # frames (b, c, d) or (d, c, b)
        #         selection = sample[1:4]
        #     elif random.random() > 0.5:
        #         # frames (b, a, d), (d, a, b), (b, e, d), or (d, e, b)
        #         selection = [sample[1], sample[0], sample[3]]
        #     else:
        #         selection = [sample[1], sample[4], sample[3]]
        #
        # images = [frames[k] for k in selection]
        # video = torch.stack([self.transforms(frame) for frame in images])
        # # NOTE: selection, e.g. [1,2,3], then images.shape = [bs, 3, 3, 128, 128]

        label = float(use_positive)
        return video, label

def get_ssd(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.sum(torch.pow(a - b, 2)).item()
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.sum(np.power(a - b, 2))
