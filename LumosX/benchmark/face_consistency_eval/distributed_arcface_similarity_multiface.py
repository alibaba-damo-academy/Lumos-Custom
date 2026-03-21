from typing import Dict, Tuple, List, Any, Union
import numpy as np
import yaml
import cv2

import torch
import torch.nn as nn
import torch.distributed
import torch.distributed as dist
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
# Make sure model code under benchmark/models is importable (yolov9, arcface, curricularface, etc.)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.general import non_max_suppression, scale_boxes
from yolov9.utils.torch_utils import smart_inference_mode
from yolov9.utils.augmentations import letterbox
from arcface.inference import build_arcface_model
from curricularface import get_model
from scipy.optimize import linear_sum_assignment


def face_crop_112x112_arcface(tensor_image: torch.Tensor, face_xyxy: List):
    np_image = tensor_image.numpy()
    cropped_image = np_image[face_xyxy[1]:face_xyxy[3], face_xyxy[0]:face_xyxy[2]]
    resized_image = cv2.resize(cropped_image, (112, 112))
    resized_image = np.transpose(resized_image, (2, 0, 1))
    tensor_image = torch.from_numpy(resized_image).float()
    tensor_image.div_(255).sub_(0.5).div_(0.5)
    return tensor_image

@smart_inference_mode()
def detect_face_crops(images, model, class_data, device):
    stride, names, pt = model.stride, model.names, model.pt
    
    imagesz = 640
    conf_thres = 0.2
    iou_thres = 0.4

    numpy_images = []
    tensor_images = []

    for image in images:
        numpy_image = image.numpy()
        numpy_images.append(numpy_image)

        image = letterbox(numpy_image, imagesz, stride=stride, auto=True)[0]
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float() / 255.0
        tensor_images.append(image)

    numpy_images = np.stack(numpy_images)
    tensor_images = torch.stack(tensor_images).to(device)

    predictions = model(tensor_images, augment=False, visualize=False)
    
    # Apply NMS
    # predictions[0][0]: (batch_size, num_class+4, prediction)
    predictions = non_max_suppression(predictions[0][0], conf_thres, iou_thres, classes=None, max_det=1000)

    # Get class name
    with open(class_data, errors='ignore') as f:
        class_id_name = yaml.safe_load(f)['names']
    
    xyxys = []
    # Process detections
    for i, prediction in enumerate(predictions):
        # scale the results
        prediction[:, :4] = scale_boxes(tensor_images[i].shape[1:], prediction[:, :4], numpy_images[i].shape).round()
        
        # Process the first face detection only
        sub_xyxy = []
        xyxy = None
        for current_prediction in prediction:
            *xyxy, conf, class_label = current_prediction
            if class_label == 0:
                xyxy = torch.stack(xyxy).cpu().numpy().reshape(1, -1)
                xyxy = xyxy[0].astype(int).tolist()

                # extend face box (0% for width, 20% for height)
                height, width, _ = numpy_images[i].shape
                xyxy[1] = max(int(xyxy[1]-0.17*(xyxy[3]-xyxy[1])), 0)
                xyxy[3] = min(int(xyxy[3]+0.03*(xyxy[3]-xyxy[1])), height-1)
                sub_xyxy.append(xyxy)
        
        xyxys.append(sub_xyxy)

    return xyxys


class DistributedArcfaceSimilarity(nn.Module):
    """
    Class for distributed arcface similarity computation
    """
    def __init__(self, yolov9_config: Dict, arcface_config: Dict):
        super().__init__()

        # Loads the pretrained face detection model
        weights = yolov9_config["weights"]
        class_data = yolov9_config["class_data"]
        self.class_data = class_data
        self.yolov9_model = DetectMultiBackend(weights, device=torch.device("cuda"), fp16=False, data=class_data).eval()

        # Loads the pretrained ArcFace model
        network = arcface_config["network"]
        weights = arcface_config["weights"]
        self.arceface_model = build_arcface_model(network, weights)
        self.arceface_model.to("cuda").eval()

        self.face_cur_model = get_model('IR_101')([112, 112])
        # self.face_cur_model.load_state_dict(torch.load("/mnt/faster_nas/dufei.df/models/ConsisID-preview/face_encoder/glint360k_curricular_face_r101_backbone.bin", map_location="cpu"))
        self.face_cur_model.load_state_dict(torch.load("/mnt/workspace/workgroup/jiazheng/workspace/LumosX/eval_code/models/face_encoder/face_encoder/glint360k_curricular_face_r101_backbone.bin", map_location="cpu"))
        self.face_cur_model = self.face_cur_model.to("cuda").eval()

        # Instantiates the mu and sigma tensors
        self.similarity_sum = torch.zeros((), dtype=torch.float64, device="cuda")
        self.similarity_sum_cur = torch.zeros((), dtype=torch.float64, device="cuda")
        self.accumulated_elements = 0

        # Waits that each process loads the feature extractor
        dist.barrier()


    def accumulate_stats(self, video: torch.Tensor, target_face_embeddings: torch.Tensor, target_face_embeddings_cur):
        """
        Accumulates the statistics
        :param video: (batch_size, frames_count, 3, height, width) tensor with videos.
        :param target_face_embeddings: (batch_size, n, 512) tensor with 512-dim arcface embeddings from n conditioning faces
        """
        # Initialize the cosine similarity and recall rate score
        current_similarity_sum = 0
        current_similarity_sum_cur = 0
        current_target_face_count = 0
        
        for current_video, current_target_face_embeddings, current_target_face_embeddings_cur in zip(video, target_face_embeddings, target_face_embeddings_cur):
            # Each frame is expected to have one frame
            current_target_face_count += len(current_video) * len(current_target_face_embeddings)
                  
            # Face detection
            face_xyxys = detect_face_crops(current_video, self.yolov9_model, self.class_data, "cuda")
            
            face_crops = []
            total_similarity = 0
            total_similarity_cur = 0
            for frame, face_xyxy in zip(current_video, face_xyxys):
                if not len(face_xyxy):
                    continue
                # Detect a face in the video frame
                face_crops = [face_crop_112x112_arcface(frame, xyxy) for xyxy in face_xyxy]
                face_crops = torch.stack(face_crops).to("cuda")
                face_embeddings = self.arceface_model(face_crops).detach().cpu()
                face_embeddings_cur = self.face_cur_model(face_crops).detach().cpu()

                similarity = cosine_similarity(
                    face_embeddings.numpy(),
                    current_target_face_embeddings.numpy()
                )

                similarity_cur = cosine_similarity(
                    face_embeddings_cur.numpy(),
                    current_target_face_embeddings_cur.numpy()
                )
            
                cost_matrix = 1 - similarity

                # 使用匈牙利算法进行匹配
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                # 提取匹配对的相似度并计算总相似度
                matched_similarities = similarity[row_indices, col_indices]
                similarity = matched_similarities.sum()
                total_similarity += similarity

                similarity_cur = similarity_cur[row_indices, col_indices].sum()
                total_similarity_cur += similarity_cur
            
            similarity = total_similarity / current_target_face_count
            similarity_cur = total_similarity_cur / current_target_face_count
            # The frames with missing face crops have similarity score zero
            current_similarity_sum = current_similarity_sum + similarity
            current_similarity_sum_cur = current_similarity_sum_cur + similarity_cur
            
        self.similarity_sum = self.similarity_sum + current_similarity_sum
        self.similarity_sum_cur = self.similarity_sum_cur + current_similarity_sum_cur

        # Updates the number of accumulated elements
        elements_count = video.shape[0]
        elements_count = torch.as_tensor(elements_count, dtype=torch.int64, device="cuda")
        # Uses distirbuted to get the current images from all processes
        torch.distributed.all_reduce(elements_count)
        self.accumulated_elements += int(elements_count.item())

    def get_statistics(self) -> torch.Tensor:
        """
        Performs computation of mu and sigma based on the accumulated statistics
        :return: () scalar tensors with the average arcface similarity and recall rate
        """

        similarity_sum = self.similarity_sum
        torch.distributed.all_reduce(similarity_sum)

        similarity_sum_cur = self.similarity_sum_cur
        torch.distributed.all_reduce(similarity_sum_cur)

        # Computes mean and covariance estimates
        similarity_sum = similarity_sum / self.accumulated_elements
        similarity_sum_cur = similarity_sum_cur / self.accumulated_elements

        return similarity_sum, similarity_sum_cur
