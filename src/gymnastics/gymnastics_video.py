import os
import pickle

import numpy as np
import torch.utils.data as data
import torch
from torchvision.datasets.video_utils import VideoClips


class GymnasticsVideo(data.Dataset):
    def __init__(self, transforms=None, train=True, test=False, count_videos=-1,
                 count_clips=-1, skip_videoframes=5, num_videoframes=100,
                 dist_videoframes=50, video_directory=None, fps=5):
        # If count_videos <= 0, use all the videos. If count_clips <= 0, use
        # all the clips from all the videos.
        self.train = train
        self.transforms = transforms
        self.video_directory = video_directory
        self.skip_videoframes = skip_videoframes
        self.num_videoframes = num_videoframes
        self.dist_videoframes = dist_videoframes

        self.video_files = sorted([
            os.path.join(video_directory, f) for f in os.listdir(video_directory) \
            if f.endswith('mp4')
        ])
        if count_videos > 0:
            self.video_files = self.video_files[:count_videos]

        clip_length_in_frames = self.num_videoframes * self.skip_videoframes
        frames_between_clips = self.dist_videoframes
        self.saved_video_clips = os.path.join(
            video_directory, 'video_clips.%dnf.%df.%ds.pkl' % (
                count_videos, clip_length_in_frames, frames_between_clips))
        if os.path.exists(self.saved_video_clips):
            print('Path Exists for video_clips: ', self.saved_video_clips)
            self.video_clips = pickle.load(open(self.saved_video_clips, 'rb'))
        else:
            print('Path does NOT exist for video_clips: ', self.saved_video_clips)            
            self.video_clips = VideoClips(
                self.video_files, clip_length_in_frames=clip_length_in_frames,
                frames_between_clips=frames_between_clips, frame_rate=fps)
            pickle.dump(self.video_clips, open(self.saved_video_clips, 'wb'))
        self.datums = self._retrieve_valid_datums(count_videos, count_clips)
        print(self.datums)
                
    def __len__(self):
        return len(self.datums)

    def _retrieve_valid_datums(self, count_videos, count_clips):
        num_clips = self.video_clips.num_clips()
        ret = []
        for flat_index in range(num_clips):
            video_idx, clip_idx = self.video_clips.get_clip_location(flat_index)
            if count_videos > 0 and video_idx >= count_videos:
                # We reached the max number of videos we want.
                break
            if count_clips > 0 and clip_idx >= count_clips:
                # We reached the max number of clips for this video.
                continue
            ret.append((flat_index, video_idx, clip_idx))

        return ret
    
    def __getitem__(self, index):
        # The video_data retrieved has shape [nf * sf, w, h, c].
        # We want to pick every sf'th frame out of that.
        flat_idx, video_idx, clip_idx = self.datums[index]
        video, _, _, _ = self.video_clips.get_clip(flat_idx)
        # video_data is [100, 360, 640, 3] --> num_videoframes, w, h, ch.
        video_data = video[0::self.skip_videoframes]
        # now video_transforms is [ch, num_videoframes, 64, 64]
        video_data = self.transforms(video_data)
        # now it's [num_videoframes, ch, 64, 64]
        video_data = torch.transpose(video_data, 0, 1)
        # path = '/misc/kcgscratch1/ChoGroup/resnick/v%d.c%d.npy' % (video_idx, clip_idx)
        # if not os.path.exists(path):
        #     np.save(path, video_data.numpy())
        return video_data, index
