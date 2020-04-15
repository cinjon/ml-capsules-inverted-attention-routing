import getpass
import os
import pickle
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
                 video_directory=None, video_names=None):
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

        video = torch.stack([self.transforms(frame) for frame in frames])
        return video, index
