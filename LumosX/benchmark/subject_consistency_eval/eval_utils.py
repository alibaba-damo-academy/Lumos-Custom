import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import os
import sys
import decord
import cv2
import base64
import io
import yaml
from fs_oss import read, ls
from tqdm import tqdm
import numpy as np

# Make sure model code under benchmark/models is importable (yolov9, arcface, etc.)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

from arcface.backbones import get_model
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.general import non_max_suppression, scale_boxes
from yolov9.utils.torch_utils import smart_inference_mode
from yolov9.utils.augmentations import letterbox
from arcface.inference import build_arcface_model

def run_example(model, processor, device, task_prompt, image_numpy, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    image = Image.fromarray(image_numpy)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer

def convert_to_od_format(data):
    """Convert Florence-2 output format to object detection format."""
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])
    return {'bboxes': bboxes, 'labels': labels}


def detect_face_crops(images, model, class_data, device):
    stride, names, pt = model.stride, model.names, model.pt
    
    imagesz = 640
    conf_thres = 0.2
    iou_thres = 0.4

    numpy_images = []
    tensor_images = []

    for image in images:
        numpy_image = image
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
    
    XYXY = []
    # Process detections
    for i, prediction in enumerate(predictions):
        # scale the results
        prediction[:, :4] = scale_boxes(tensor_images[i].shape[1:], prediction[:, :4], numpy_images[i].shape).round()
        
        # Process the first face detection only
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
                break

        XYXY.append(xyxy)

    return XYXY[0]


def inference(net, device, numpy_img, face_xyxy=None):
    if face_xyxy:
        img = numpy_img[face_xyxy[1]:face_xyxy[3], face_xyxy[0]:face_xyxy[2]]
    else:
        img = numpy_img
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    img.div_(255).sub_(0.5).div_(0.5)
    feat = net(img)
    return feat

def owlv2_detect(model, processor, device, image_numpy, text):
    image = Image.fromarray(image_numpy)
    text_labels = [[text]]
    inputs = processor(text=text_labels, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    # target_sizes = torch.tensor([(image.height, image.width)])
    W, H = image.size[0], image.size[1]
    target_sizes = torch.Tensor([[max(W,H), max(W,H)]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
    )
    # Retrieve predictions for the first image for the corresponding text queries
    result = results[0]
    # boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
    return result

def valid_bbox(box):
    return box[3] > box[1] and box[2] > box[0]