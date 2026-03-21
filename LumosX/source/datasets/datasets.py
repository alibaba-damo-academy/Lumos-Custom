import os
from glob import glob
import json
import math
import numpy as np
from pandas.core.missing import F
import torch
from PIL import ImageFile, Image
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
import pandas as pd
from source.registry import DATASETS
from copy import deepcopy
from .read_video import read_video
from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop
from .fs import read_pil_image, read
import random
from tqdm import tqdm
from pathlib import Path
import decord
import cv2

decord.bridge.set_bridge("torch")

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FPS = 120


@DATASETS.register_module()
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


@DATASETS.register_module()
class VideoAlchemyDataset(VideoTextDataset):
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
        self.mask_base_path = "/mnt_huaniu/Data/VideoAlchemyFiles_v2"
        # self.data['height'] = self.data['height'] * 4
        # self.data['width'] = self.data['width'] * 4
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

    def add_white_background(self, np_image):
        background_mask = np_image[:, :, 3] == 0
        np_image[background_mask] = [255, 255, 255, 255]
        np_image = np_image[:, :, :3]
        return np_image

    def load_img_masks(self, masks_dirs, original_imgs,  video_class_dir, selected_frames):
        mask_imgs = list()
        for mask_dir in masks_dirs:
            mask_img_path = os.path.join(video_class_dir, mask_dir)
            mask_img = np.array(Image.open(mask_img_path))
            mask_img_normalized = mask_img.astype(np.float32) / 255.0
            
            idx = int(mask_dir.split('_')[-1][5:-4])
            position= selected_frames.index(idx)
            image = original_imgs[position]
            alpha_channel = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
            rgba_image = np.concatenate((image, alpha_channel), axis=-1)
            rgba_image[...,-1] = rgba_image[...,-1] *  mask_img_normalized
            out = self.add_white_background(rgba_image)

            ######## crop mask ##########
            mask_crop = cv2.inRange(out, (255, 255, 255), (255, 255, 255))
            non_white = cv2.bitwise_not(mask_crop)
            contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                out_image = out 
            else:
                all_points = np.concatenate(contours)
                x, y, w, h = cv2.boundingRect(all_points)
                out_image =out[y:y+h, x:x+w]

            mask_imgs.append(out_image)
        return mask_imgs

    def load_face_crops(self, human_crops, images, selected_frames):
        face_imgs = list()
        for face_key, face_value in human_crops.items():
            face_key = int(face_key)
            position= selected_frames.index(face_key)
            np_image = images[position]
            
            x0, y0, x1, y1 = face_value
            np_image = np_image[y0:y1, x0:x1]
            face_imgs.append(np_image)
        return face_imgs

        # mask_img = np.array(Image.open('/mnt/workspace/workgroup/jiazheng/workspace/VidGen-master/mask_dir/4448a5ba8ff8a7b78265c011826dfea0_0:00:07.666667_0:00:18.666667/black_cap_frame16.png'))
        # mask_img_normalized = mask_img.astype(np.float32) / 255.0
        # image = images[0]
        # alpha_channel = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        # rgba_image = np.concatenate((image, alpha_channel), axis=-1)
        # rgba_image[...,-1] = rgba_image[...,-1] *  mask_img_normalized
        # out = self.add_white_background(rgba_image)
        # Image.fromarray(rgba_image, "RGBA").save('2.png')
        # Image.fromarray(out, "RGB").save('3.png')
    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, duration, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        if 'path' not in sample:
            key = sample["key"]
            with self.env.begin(write=False) as txn:
                info = pickle.loads(txn.get(key.encode("ascii")))
            path, text = info
        else:
            path = sample["path"].replace("/mnt/sh_nas/moyuan.yty/", "/home/dufei.df/huaniu_workspace/")
            path = path.replace('oss://', '/ossfs/')
        # path = path.replace("oss://", "/root/")
        video_name = path.split('/')[-1][:-4]
        mask_dir_path = os.path.join(self.mask_base_path, path.split('/')[3], path.split('/')[4], path.split('/')[5], path.split('/')[6], video_name) 
        # mask_dir_path = os.path.join(self.mask_base_path, video_name)
        if os.path.isdir(mask_dir_path):
            info_path = os.path.join(mask_dir_path, 'info.json')
            with open(info_path, "r", encoding="utf-8") as file:
                info_json = json.load(file)
            if 'person' in info_json["word_tags_path"].keys():
                persons_info = info_json["word_tags_path"]['person']
            else:
                persons_info = None
            if 'other_subjects' in info_json["word_tags_path"].keys():
                other_subjects_info = info_json["word_tags_path"]['other_subjects']
            else:
                other_subjects_info = None
            selected_frames = info_json['selected_frames']
            video_file = info_json['video']
            # video_class_dir = video_file.split('/')[-1][:-4]
            video_reader = decord.VideoReader(video_file)
            images = [video_reader[idx].numpy() for idx in selected_frames]

            mask_dict = dict()
            transform_img = get_transforms_image(self.transform_name, (height, width))
            transform_img_bg = get_transforms_image('resize_crop_bg', (height, width))
            #######################background
            if 'background' in info_json['word_tags_path'].keys():
                background_name = [key for key in info_json['word_tags_path']['background'].keys()][0]
                if len(info_json['word_tags_path']['background'][background_name]) ==3:
                    num = 1
                else:
                    num = 0
                background_path = os.path.join(mask_dir_path, info_json['word_tags_path']['background'][background_name][num])
                background_img = [transform_img_bg(Image.open(background_path))]
                # background_img = [transform_img(Image.open(background_path))]
                mask_dict[background_name] = background_img
            # else:
            #     background_img = [torch.zeros(3, height, width)] # TODO ones
            #     mask_dict['no_bg'] = background_img

            if persons_info is not None:
                ratio_subject = 0.7 #TODO
                random_number = random.random()
                transform_img_others = get_transforms_image('resize_crop_others', (height, width))
                human_num = 0
                for human_i, subject_human in enumerate(persons_info):
                    for human_key, human_vale in subject_human['subject'].items():
                        if human_key == 'face':
                            continue
                        else:
                            if random_number < 0.7:
                                if 'face' in subject_human['subject'].keys():
                                    human_dict = dict()
                                    human_crops = subject_human['subject']['face']
                                    # mask_human_imgs =  self.load_img_masks(human_vale, images, mask_dir_path, selected_frames)
                                    crop_face_imgs = self.load_face_crops(human_crops, images, selected_frames)
                                    mask_key = ' '.join(human_vale[0].split('_')[:-1])
                                    # image_pil = Image.fromarray(mask_human_imgs[0], mode="RGB")
                                    # image_pils = [Image.fromarray(mask_human_img, mode="RGB") for mask_human_img in mask_human_imgs]
                                    image_pils = [Image.fromarray(crop_face_img, mode="RGB") for crop_face_img in crop_face_imgs]
                                    image_tensors = [transform_img_others(image_pil) for image_pil in image_pils]
                                    # image_tensors = [transform_img(image_pil) for image_pil in image_pils]
                                    human_dict[human_key] = image_tensors
                                    # mask_dict[human_key] = image_tensors

                                    # image_tensor = image_tensors[0]
                                    # image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
                                    # image_np = image_tensor.permute(1, 2, 0).numpy()  # First adjust dimensions
                                    # image_np = (image_np * 255).astype(np.uint8)
                                    # image_pil = Image.fromarray(image_np)
                                    # image_pil.save('output_image1.png')
                                    if 'attributes' in subject_human.keys():
                                        for human_attributes in subject_human['attributes']:                
                                            for human_attributes_key, human_attributes_vale in human_attributes.items():
                                                mask_human_attributes_imgs = self.load_img_masks(human_attributes_vale, images, mask_dir_path, selected_frames)
                                                # image_pil = Image.fromarray(mask_human_imgs[0], mode="RGB")
                                                image_pils = [Image.fromarray(mask_human_attributes_img, mode="RGB") for mask_human_attributes_img in mask_human_attributes_imgs]
                                                image_tensors = [transform_img_others(image_pil) for image_pil in image_pils]
                                                # image_tensors = [transform_img(image_pil) for image_pil in image_pils]

                                                human_dict[human_attributes_key] = image_tensors
                                    human_idx = 'human_x_' + str(human_num)
                                    # assert len(human_dict) != 0
                                    mask_dict[human_idx] = human_dict
                                    human_num += 1
                                    # mask_dict[human_attributes_key] = image_tensors
                            else:
                                mask_human_imgs =  self.load_img_masks(human_vale, images, mask_dir_path, selected_frames)
                                mask_key = ' '.join(human_vale[0].split('_')[:-1])
                                # image_pil = Image.fromarray(mask_human_imgs[0], mode="RGB")
                                image_pils = [Image.fromarray(mask_human_img, mode="RGB") for mask_human_img in mask_human_imgs]
                                image_tensors = [transform_img_others(image_pil) for image_pil in image_pils]
                                mask_dict[human_key] = image_tensors
                        
            if other_subjects_info is not None:
                for other_subject in other_subjects_info:
                    for other_subject_key, other_subject_vale in other_subject.items():
                        other_subject_imgs = self.load_img_masks(other_subject_vale, images, mask_dir_path, selected_frames)
                        # image_pil = Image.fromarray(mask_human_imgs[0], mode="RGB")
                        image_pils = [Image.fromarray(other_subject_img, mode="RGB") for other_subject_img in other_subject_imgs]
                        image_tensors = [transform_img_others(image_pil) for image_pil in image_pils]
                        # image_tensors = [transform_img(image_pil) for image_pil in image_pils]
                        mask_dict[other_subject_key] = image_tensors

                
                # image_tensor = mask_dict['workbench'][0]
                # image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
                # image_np = image_tensor.permute(1, 2, 0).numpy()  # 先调整维度
                # image_np = (image_np * 255).astype(np.uint8)
                # image_pil = Image.fromarray(image_np)
                # image_pil.save('output_image0.png')


        if not mask_dict:
            raise ValueError("mask dict is empty")
        file_type = self.get_type(path)
        ar = height / width

        if file_type == "video":
            video_fp = read(path)
            video_reader = decord.VideoReader(video_fp)
            h, w = video_reader[0].shape[:2]
            rh, rw = height / h, width / w
            if rh > rw:
                sh, sw = height, round(w * rh)
            else:
                sh, sw = round(h * rw), width
            del video_reader

            video_fp = read(path)
            video_reader = decord.VideoReader(video_fp, width=sw, height=sh)

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

            video_fps = self.sample_fps

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            if path.startswith('oss'):
                image = read_pil_image(path)
            else:
                image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)
            num_frames = 1
            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        
        # assert len(mask_dict) == 1 #TODO
        ret = {
            "path": path,
            "video": video,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
            "mask_dict": mask_dict,
        }
        if self.get_text:
            ret["text"] = sample["text"]
        else:
            ret["text"] = text
        
        # ret["mask_dict"] = [mask_dict]

        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index):
        from training_acc.dist.parallel_state import is_enable_sequence_parallel
        if not is_enable_sequence_parallel():
            while True:
                try:
                    return self.getitem(index)
                except Exception as e:
                    print("--------------------------------- read data error: ", e)
                    index, duration, height, width = [int(val) for val in index.split("-")]
                    #index, duration, height, width
                    if duration == 0: # Image
                        indices = self.data[self.data['duration'] == duration].index
                    else:
                        indices = self.data[self.data['duration'] >= duration].index
                        if len(indices) == 0: # Avoid cases where there's no video with corresponding duration, may be less
                            indices = self.data[self.data['duration'] != 0].index
                    random_index = np.random.choice(indices) # This random may affect randomness across different cards, but this should be the case, as long as operations on a single rank are consistent, just set the seed properly
                    index = f"{random_index}-{duration}-{height}-{width}"
        else:
            valid = torch.ones(1).bool()
            data = {}

            try:                
                data = self.getitem(index)
            except Exception as e:
                valid = torch.zeros(1).bool()
                
                from training_acc.logger import logger
                from training_acc.dist import log_rank
                logger.info(log_rank(f"--------------------------------- read data index:{index}, error: {e}"))
                
            data["valid"] = valid
            return data


@DATASETS.register_module()
class VariableVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        data_path=None,
        num_frames=None,
        frame_interval=1,
        image_size=(None, None),
        transform_name=None,
        dummy_text_feature=False,
        **kwargs
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, transform_name=None)
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))
        # self.data['height'] = self.data['height'] * 4
        # self.data['width'] = self.data['width'] * 4
        self.dummy_text_feature = dummy_text_feature

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        path = sample["path"].replace("/mnt/sh_nas/moyuan.yty/", "/home/dufei.df/huaniu_workspace/")
        file_type = self.get_type(path)
        ar = height / width

        # sample_fps = 24  # default fps
        if file_type == "video":
            # loading
            # vframes, vinfo = read_video(path, backend="av")
            # video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # # Sampling video frames
            # video = temporal_random_crop(vframes, num_frames, self.frame_interval)
            
            # video = video.clone()
            # del vframes
            video_fp = read(path)
            video_reader = decord.VideoReader(video_fp)
            video_fps = video_reader.get_avg_fps()
            ori_video_length = len(video_reader)
            # normed_video_length = round(ori_video_length / ori_fps * self.sample_fps)
            
            start_idx = np.random.randint(0, ori_video_length - num_frames * self.frame_interval)
            batch = np.arange(start_idx, start_idx + num_frames * self.frame_interval, self.frame_interval)
            video = video_reader.get_batch(batch).permute(0, 3, 1, 2)

            video_fps = video_fps // self.frame_interval

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            if path.startswith('oss'):
                image = read_pil_image(path)
            else:
                image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)
            

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        ret = {
            "video": video,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }
        if self.get_text:
            ret["text"] = sample["text"]
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            print("--------------------------------- read data error: ", e)
            return None


@DATASETS.register_module()
class VariableVideoTextWithDurationDataset(VideoTextDataset):
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
        # self.data['height'] = self.data['height'] * 4
        # self.data['width'] = self.data['width'] * 4
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

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, duration, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        if 'path' not in sample:
            key = sample["key"]
            with self.env.begin(write=False) as txn:
                info = pickle.loads(txn.get(key.encode("ascii")))
            path, text = info
        else:
            path = sample["path"].replace("/mnt/sh_nas/moyuan.yty/", "/home/dufei.df/huaniu_workspace/")
            path = path.replace('oss://', '/ossfs/')
        # path = path.replace("oss://", "/root/")
        file_type = self.get_type(path)
        ar = height / width

        if file_type == "video":
            video_fp = read(path)
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

            video_fps = self.sample_fps

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            if path.startswith('oss'):
                image = read_pil_image(path)
            else:
                image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)
            num_frames = 1
            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        ret = {
            "video": video,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }
        if self.get_text:
            ret["text"] = sample["text"]
        else:
            ret["text"] = text
        
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index):
        from training_acc.dist.parallel_state import is_enable_sequence_parallel
        if not is_enable_sequence_parallel():
            while True:
                try:
                    return self.getitem(index)
                except Exception as e:
                    print("--------------------------------- read data error: ", e)
                    index, duration, height, width = [int(val) for val in index.split("-")]
                    #index, duration, height, width
                    if duration == 0: # Image
                        indices = self.data[self.data['duration'] == duration].index
                    else:
                        indices = self.data[self.data['duration'] >= duration].index
                        if len(indices) == 0: # Avoid cases where there's no video with corresponding duration, may be less
                            indices = self.data[self.data['duration'] != 0].index
                    random_index = np.random.choice(indices) # This random may affect randomness across different cards, but this should be the case, as long as operations on a single rank are consistent, just set the seed properly
                    index = f"{random_index}-{duration}-{height}-{width}"
        else:
            valid = torch.ones(1).bool()
            data = {}

            try:                
                data = self.getitem(index)
            except Exception as e:
                valid = torch.zeros(1).bool()
                
                from training_acc.logger import logger
                from training_acc.dist import log_rank
                logger.info(log_rank(f"--------------------------------- read data index:{index}, error: {e}"))
                
            data["valid"] = valid
            return data
                    

@DATASETS.register_module()
class BatchFeatureDataset(torch.utils.data.Dataset):
    """
    The dataset is composed of multiple .bin files.
    Each .bin file is a list of batch data (like a buffer). All .bin files have the same length.
    In each training iteration, one batch is fetched from the current buffer.
    Once a buffer is consumed, load another one.
    Avoid loading the same .bin on two difference GPUs, i.e., one .bin is assigned to one GPU only.
    """

    def __init__(self, data_path=None):
        self.path_list = sorted(glob(data_path + "/**/*.bin"))

        self._len_buffer = len(torch.load(self.path_list[0]))
        self._num_buffers = len(self.path_list)
        self.num_samples = self.len_buffer * len(self.path_list)

        self.cur_file_idx = -1
        self.cur_buffer = None

    @property
    def num_buffers(self):
        return self._num_buffers

    @property
    def len_buffer(self):
        return self._len_buffer

    def _load_buffer(self, idx):
        file_idx = idx // self.len_buffer
        if file_idx != self.cur_file_idx:
            self.cur_file_idx = file_idx
            self.cur_buffer = torch.load(self.path_list[file_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._load_buffer(idx)

        batch = self.cur_buffer[idx % self.len_buffer]  # dict; keys are {'x', 'fps'} and text related
        ret = {
            "video": batch["x"],
            "text": batch["y"],
            "mask": batch["mask"],
            "fps": batch["fps"],
            "height": batch["height"],
            "width": batch["width"],
            "num_frames": batch["num_frames"],
        }
        return ret

@DATASETS.register_module()
class VariableVideoTextPerRankDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path=None,
        image_data_path=None,
        rank=None,
        num_frames=None,
        frame_interval=1,
        image_size=(None, None),
        transform_name=None,
        dummy_text_feature=False,
        sample_fps = 16,
        image_percent = None,
        add_one=False,
        **kwargs
    ):
        self.rank = rank
        assert data_path is not None
        data = read_file(data_path.format(int(rank)))
        if image_data_path is not None:
            image_data = read_file(image_data_path.format(int(rank)))
            self.data = pd.concat([data, image_data]).reset_index(drop=True)
        else:
            self.data = data
        self.bucket_id_list = sorted(self.data['bucket_id'].drop_duplicates().to_list())
        self.bucket_id_num_count = self.data['bucket_id'].value_counts()
        self.image_percent = image_percent
        self.total_lens = len(self.data)
        
        # self.data = self.data[self.data['path'].str.endswith(".mp4")][:100]
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }
        self.transform_name = transform_name
        # self.data['height'] = self.data['height'] * 4
        # self.data['width'] = self.data['width'] * 4
        self.dummy_text_feature = dummy_text_feature
        self.sample_fps = sample_fps
        self.add_one = add_one
    
    def __len__(self):
        return self.total_lens
    
    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"
        
    def getitem(self, index):        
        sample = deepcopy(self.data.iloc[index])
        
        bucket_id, path, text = sample['bucket_id'], sample['path'], sample['text']
        
        duration, height, width, _ = [int(val) for val in bucket_id.split("-")]
        
        path = path.replace("/mnt/sh_nas/moyuan.yty/", "/home/dufei.df/huaniu_workspace/")
        # path = path.replace("oss://", "/root/")
        file_type = self.get_type(path)
        ar = height / width
        
        if file_type == "video":
            video_fp = read(path)
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

            video_fps = self.sample_fps

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            if path.startswith('oss'):
                image = read_pil_image(path)
            else:
                image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)
            num_frames = 1
            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        ret = {
            "video": video,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
            "text": text
        }
        
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index): 
        from training_acc.dist.parallel_state import is_enable_sequence_parallel
        if not is_enable_sequence_parallel():
            while True:
                try:
                    return self.getitem(index)
                except Exception as e:
                    print("--------------------------------- read data error: ", e)
                    sample = deepcopy(self.data.iloc[index])
                    bucket_id = sample['bucket_id']
                    duration, height, width, _ = [int(val) for val in bucket_id.split("-")]        

                    if self.bucket_id_num_count[bucket_id] == 1:
                        candidate_bucket_id = []
                        for k in self.bucket_id_list:
                            sub_duration = int(k.split("-")[0])
                            if duration == 0 and sub_duration != 0:
                                continue
                            elif subduration < duration:
                                continue
                            candidate_bucket_id.append(k)
                        bucket_id = np.random.choice(candidate_bucket_id)
                    index =  np.random.choice(self.data.index[self.data['bucket_id'] == bucket_id].tolist())
        else:
            valid = torch.ones(1).bool()
            data = {}

            try:                
                data = self.getitem(index)
            except Exception as e:
                valid = torch.zeros(1).bool()
                
                from training_acc.logger import logger
                from training_acc.dist import log_rank
                logger.info(log_rank(f"--------------------------------- read data index:{index}, error: {e}"))
                
            data["valid"] = valid
            return data