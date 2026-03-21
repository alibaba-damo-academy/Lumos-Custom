import numpy as np
from typing import Dict, Tuple, List, Any, Union

import torch
import torch.nn as nn
import torch.distributed
import torch.distributed as dist

import os
import sys

# Make sure RAFT code under benchmark/models is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from raft.core.raft import RAFT
from raft.core.utils_core.utils import InputPadder


class DistributedDynamicDegree(nn.Module):
    """
    Class for distributed Dynamic Degree score computation
    """
    def __init__(self, raft_config: Dict):
        super().__init__()

        # Loads the pretrained RAFT model
        self.raft_model = RAFT(raft_config)
        checkpoint = torch.load(raft_config.model, map_location="cpu")
        unwarpped_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.raft_model.load_state_dict(unwarpped_checkpoint)
        self.raft_model.to("cuda")
        self.raft_model.eval()

        # Instantiates the mu and sigma tensors
        self.score_sum = torch.zeros((), dtype=torch.float64, device="cuda")
        self.accumulated_elements = 0

        # Waits that each process loads the feature extractor
        dist.barrier()

    def get_score(self, img, flo):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()

        u = flo[:,:,0]
        v = flo[:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = int(h*w*0.05)

        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])

        return max_rad.item()

    def set_params(self, frame, count):
        scale = min(list(frame.shape)[-2:])
        self.params = {"thres":6.0*(scale/256.0), "count_num":round(4*(count/16.0))}

    def extract_frame(self, video, interval=1):
        extracted_frames = []
        for i in range(0, len(video), interval):
            extracted_frames.append(video[i])
        return extracted_frames

    def check_move(self, score_list):
        thres = self.params["thres"]
        count_num = self.params["count_num"]
        count = 0
        for score in score_list:
            if score > thres:
                count += 1
            if count >= count_num:
                return True
        return False
    
    def accumulate_stats(self, video: torch.Tensor, video_framerate: List):
        """
        Accumulates the statistics
        :param video: (batch_size, frames_count, 3, height, width) tensor with videos.
        """
        # downsample video to framerate 8
        interval = round(video_framerate/8)
        video = video.to("cuda")
        video = self.extract_frame(video, interval)
    
        with torch.no_grad():
            # set parameters for thresholding
            thres = 6.0 * (min(list(video[0].shape)[-2:]) / 256.0)
            count = round(len(video) / 4.0)
            self.params = {"thres" : thres, "count_num": count}
        
            # run inference
            static_score = []
            for image1, image2 in zip(video[:-1], video[1:]):
                image1 = image1.unsqueeze(0)
                image2 = image2.unsqueeze(0)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.raft_model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(image1, flow_up)
                static_score.append(max_rad)
            
            # parse results
            whether_move = self.check_move(static_score)

        # Updates the total score
        self.score_sum = self.score_sum + whether_move

        # Updates the number of accumulated elements
        elements_count = 1
        elements_count = torch.as_tensor(elements_count, dtype=torch.int64, device="cuda")
        # Uses distirbuted to get the current images from all processes
        torch.distributed.all_reduce(elements_count)
        self.accumulated_elements += int(elements_count.item())

    def get_statistics(self) -> torch.Tensor:
        """
        Performs computation of mu and sigma based on the accumulated statistics
        :return: () scalar tensor with the average cosine similarity
        """

        score_sum = self.score_sum
        torch.distributed.all_reduce(score_sum)
        
        # Computes mean and covariance estimates
        score_sum = score_sum / self.accumulated_elements

        return score_sum
