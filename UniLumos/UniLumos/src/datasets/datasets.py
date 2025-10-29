import os
import math
import numpy as np
import torch
import decord
from PIL import ImageFile
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader


from .read_video import read_video
from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop

decord.bridge.set_bridge("torch")

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FPS = 120


class VideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path=None,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="center",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        # self.data = self.data[self.data['path'].str.endswith(".mp4")][:100]
        self.get_text = "text" in self.data.columns
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)

        if file_type == "video":
            # loading
            vframes, vinfo = read_video(path, backend="av")
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)

            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        ret = {"video": video, "fps": video_fps}
        if self.get_text:
            ret["text"] = sample["text"]
        return ret

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


class LumosDataset(VideoTextDataset):
    def __init__(
        self,
        data_path=None,
        lmdb_path=None,
        num_frames=None,
        frame_interval=1,
        image_size=(None, None),
        transform_name=None,
        dummy_text_feature=False,
        sample_fps = 16,
        add_one=False,
        **kwargs
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, transform_name=None)
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))
        self.dummy_text_feature = dummy_text_feature
        self.sample_fps = sample_fps
        self.add_one = add_one
        if lmdb_path is not None:
            self.env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            # self.txn = self.env.begin(buffers=True, write=False)

    def get_data_info(self, index):
        T = self.data.iloc[index]["duration"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def get_videos(self, path, duration, height, width):
        
        video_fp = path
        video_reader = decord.VideoReader(video_fp)
        ori_fps = video_reader.get_avg_fps()
        ori_video_length = len(video_reader)
        
        num_frames = int(duration * self.sample_fps)
        if self.add_one:
            num_frames = num_frames + 1
        
        required_len = math.ceil(num_frames / self.sample_fps * ori_fps)
        clip_len = min(10, max(ori_video_length - required_len, 0) // 2)
        normed_video_length = round((ori_video_length - 2*clip_len) / ori_fps * self.sample_fps)
        normed_video_length = max(num_frames, normed_video_length)
        batch_index_all = np.linspace(clip_len, ori_video_length - 1 - clip_len, normed_video_length).round().astype(int)
        start_idx = 0 #random.randint(0, normed_video_length - num_frames)
        batch_index = batch_index_all[start_idx:start_idx+num_frames]
        
        video = video_reader.get_batch(batch_index).permute(0, 3, 1, 2)
        # transform
        transform = get_transforms_video(self.transform_name, (height, width))
        video = transform(video)  # T C H W

        return video

    def get_each_video(self, path, deg_path, bg_path, duration, height, width):
        
        # video_real
        video_real = self.get_videos(path, duration, height, width)
        video_name = os.path.splitext(os.path.basename(path))[0]
        path_deg = deg_path
        path_bg = bg_path

        video_deg  = self.get_videos(path_deg, duration, height, width)
        video_bg   = self.get_videos(path_bg, duration, height, width)

        return video_real, video_deg, video_bg

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        # index, duration, height, width = [int(val) for val in index.split("-")]
        duration = 4
        height = 480
        width = 832

        sample = self.data.iloc[index]

        path = sample["path"]
        deg_path = sample["deg_path"]
        bg_path = sample["bg_path"]
        text = sample["text"]

        file_type = self.get_type(path)
        ar = height / width

        if file_type == "video":
            video_real, video_deg, video_bg = self.get_each_video(path, deg_path, bg_path, duration, height, width)

        
        num_frames = int(duration * self.sample_fps)
        if self.add_one:
            num_frames = num_frames + 1
        video_fps = self.sample_fps

        # TCHW -> CTHW
        video_real = video_real.permute(1, 0, 2, 3)
        video_deg = video_deg.permute(1, 0, 2, 3)
        video_bg = video_bg.permute(1, 0, 2, 3)

        ret = {
            "video": video_real,
            "video_deg": video_deg,
            "video_bg": video_bg,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
            "bg_path": bg_path,
            "ori_path": path,
        }

        ret["text"] = text
        
        return ret

    def __getitem__(self, index):
        
        return self.getitem(index)
                    