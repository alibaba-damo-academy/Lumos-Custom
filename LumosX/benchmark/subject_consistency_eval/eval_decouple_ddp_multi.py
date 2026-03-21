import torch
import torch.distributed as dist
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForCausalLM, Owlv2Processor, Owlv2ForObjectDetection,
    CLIPModel, AutoTokenizer, AutoImageProcessor, AutoModel
)
import os
import sys
import decord
import numpy as np
import glob
import argparse
from tqdm import tqdm
from fs_oss import ls

# Make sure model code under benchmark/models is importable (arcface, curricularface, yolov9, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from arcface.backbones import get_model
from curricularface import get_model as get_model_currface
from yolov9.models.common import DetectMultiBackend
from eval_utils import (
    run_example, convert_to_od_format, detect_face_crops, inference,
    owlv2_detect, valid_bbox
)


def init_dist(launcher="pytorch", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)    
    return local_rank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decoupled evaluation script for subject consistency metrics"
    )
    parser.add_argument("--video_dir", type=str, required=True, help="Path to generated videos directory")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to reference images directory")
    parser.add_argument("--florence_path", type=str, default="../models/florence2-large-ft", help="Path to Florence-2 model")
    parser.add_argument("--yolo_weights", type=str, default="yolov9/weight/best.pt", help="Path to YOLOv9 weights")
    parser.add_argument("--yolo_config", type=str, default="yolov9/data/coco.yaml", help="Path to YOLOv9 config")
    parser.add_argument("--arcface_weights", type=str, default="arcface/weight/backbone.pth", help="Path to ArcFace weights")
    parser.add_argument("--curricular_face_weights", type=str, default="../models/face_encoder/face_encoder/glint360k_curricular_face_r101_backbone.bin", help="Path to CurricularFace weights")
    parser.add_argument("--owlv2_path", type=str, default="../models/owlv2-base-patch16-ensemble", help="Path to OWLv2 model")
    parser.add_argument("--clip_path", type=str, default="../models/clip-vit-large-patch14", help="Path to CLIP model")
    parser.add_argument("--dinov2_path", type=str, default="../models/dinov2-base", help="Path to DINOv2 model")
    args = parser.parse_args()

    # 初始化分布式环境
    local_rank = init_dist()
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    # 设置log文件（仅在主进程）
    log_file = None
    log_name = None
    if is_main_process:
        # 从video_dir中提取最后一部分作为文件名
        video_dir_parts = args.video_dir.rstrip('/').split('/')
        log_name = video_dir_parts[-1] if video_dir_parts[-1] else video_dir_parts[-2]
        
        # 创建results目录
        os.makedirs("./results", exist_ok=True)
        
        # 创建log文件
        log_file_path = os.path.join("./results", f"{log_name}.log")
        log_file = open(log_file_path, 'w', encoding='utf-8')
        
        # 写入log抬头
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Evaluation Results for: {log_name}\n")
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Video Directory: {args.video_dir}\n")
        log_file.write(f"Reference Image Directory: {args.ref_image}\n")
        log_file.write(f"Total Processes: {num_processes}\n")
        log_file.write(f"{'='*60}\n\n")
        log_file.flush()
    
    # 创建同时输出到控制台和文件的打印函数
    def log_print(*args, **kwargs):
        """同时打印到控制台和log文件"""
        if is_main_process:
            # 获取end参数，默认为'\n'
            end = kwargs.get('end', '\n')
            # 先打印到控制台
            print(*args, **kwargs)
            # 然后写入文件
            if log_file is not None:
                # 将print的内容转换为字符串
                output = ' '.join(str(arg) for arg in args)
                log_file.write(output + end)
                log_file.flush()

    device = f"cuda:{local_rank}"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_florence = AutoModelForCausalLM.from_pretrained(args.florence_path, torch_dtype=torch_dtype, trust_remote_code=True).eval().to(device)
    processor_florence = AutoProcessor.from_pretrained(args.florence_path, trust_remote_code=True)
    task_prompt = '<OPEN_VOCABULARY_DETECTION>'
    if is_main_process:
        log_print("Load model Florence-2.")
    
    yolov9_model = DetectMultiBackend(args.yolo_weights, device=torch.device(device), fp16=False, data=None).eval()
    if is_main_process:
        log_print("Load model YOLO.")

    net_arcface = get_model('r100', fp16=False).to(device)
    net_arcface.load_state_dict(torch.load(args.arcface_weights, map_location="cpu", weights_only=False))
    net_arcface.eval()
    if is_main_process:
        log_print("Load model Arcface.")

    face_cur_model = get_model_currface('IR_101')([112, 112])
    face_cur_model.load_state_dict(torch.load(args.curricular_face_weights, map_location="cpu", weights_only=False))
    face_cur_model = face_cur_model.to(device).eval()
    if is_main_process:
        log_print("Load model CurricularFace.")

    processor_owlv2 = Owlv2Processor.from_pretrained(args.owlv2_path)
    model_owlv2 = Owlv2ForObjectDetection.from_pretrained(args.owlv2_path).eval().to(device)
    if is_main_process:
        log_print("Load model Owlv2.")

    model_clip = CLIPModel.from_pretrained(args.clip_path).eval().to(device)
    processor_clip = AutoProcessor.from_pretrained(args.clip_path)
    tokenizer_clip = AutoTokenizer.from_pretrained(args.clip_path)
    if is_main_process:
        log_print("Load model CLIP.")

    processor_dinov2 = AutoImageProcessor.from_pretrained(args.dinov2_path)
    model_dinov2 = AutoModel.from_pretrained(args.dinov2_path).eval().to(device)
    if is_main_process:
        log_print("Load model DINO.")

    test_path = args.ref_image
    test_video_dir = args.video_dir
    files = ls(test_path)
    
    # 数据并行：将文件分配给不同的进程
    if len(files) % num_processes != 0:
        files = files + files[:(num_processes - len(files) % num_processes)]
    files = files[local_rank::num_processes]
    
    # 按person分组收集结果
    res_by_person = {
        'person1': {'face_sim': [], 'curr_face_sim': [], 'clip_t': [], 'clip_i': [], 'dino_i': []},
        'person2': {'face_sim': [], 'curr_face_sim': [], 'clip_t': [], 'clip_i': [], 'dino_i': []},
        'person3': {'face_sim': [], 'curr_face_sim': [], 'clip_t': [], 'clip_i': [], 'dino_i': []},
        'overall': {'face_sim': [], 'curr_face_sim': [], 'clip_t': [], 'clip_i': [], 'dino_i': []}
    }
    
    def extract_person_from_video_name(video_name):
        """从视频文件名中提取person信息"""
        video_name_lower = video_name.lower()
        if 'person1' in video_name_lower or 'video_person1' in video_name_lower:
            return 'person1'
        elif 'person2' in video_name_lower or 'video_person2' in video_name_lower:
            return 'person2'
        elif 'person3' in video_name_lower or 'video_person3' in video_name_lower:
            return 'person3'
        else:
            return 'overall'  # 如果无法识别，归为overall
    
    for item in tqdm(files, disable=not is_main_process):
        video_name = f"{item}*.mp4"
        item_path = os.path.join(test_path, item)
        image_path = os.path.join(item_path, "image.png")
        video_path = glob.glob(os.path.join(test_video_dir, video_name))[0]
        # 从视频路径中提取person信息
        video_basename = os.path.basename(video_path)
        person_key = extract_person_from_video_name(video_basename)
        subjects = ls(os.path.join(item_path, "subjects"))
        tmp_face_sim = []
        tmp_curr_face_sim = []
        tmp_clip_t = []
        tmp_clip_i = []
        tmp_dino_i = []
        
        vr = decord.VideoReader(video_path)
        video_length = len(vr)
        image_numpy = vr[video_length//2].asnumpy()

        sub_clip_t = []
        sub_clip_i = []
        sub_dino_i = []
        for sub in subjects:
            sub_dir = os.path.join(item_path, "subjects", sub)
            _all_sub = ls(sub_dir)
            person = [item for item in _all_sub if "face_" in item][0].split("face_", 1)[1].replace('.png', '')
            attrs = [item.replace(".png", "") for item in _all_sub if ("face_" not in item and "fullfullbody" not in item)]
            text = person + " with " + " and ".join(attrs)
            # 匹配
            with torch.no_grad():
                results = run_example(model_florence, processor_florence, device, task_prompt, image_numpy, text_input=text)
            bbox_results = convert_to_od_format(results[task_prompt])['bboxes']
            if len(bbox_results) > 0:
                bbox = bbox_results[0]
                bbox = [round(coord) for coord in bbox]
                H, W, _ = image_numpy.shape
                bbox = [int(np.clip(bbox[0], 0, W)), int(np.clip(bbox[1], 0, H)), int(np.clip(bbox[2], 0, W)), int(np.clip(bbox[3], 0, H))]
                bbox = bbox if valid_bbox(bbox) else None
            else:
                bbox = None
                sub_clip_t.append(0)
                sub_clip_i.append(0)
                sub_dino_i.append(0)
                tmp_face_sim.append(0)
                continue
            
            image_sub_numpy = image_numpy[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else image_numpy
            face_gt_path = os.path.join(item_path, "subjects", sub, [item for item in _all_sub if "face_" in item][0])
            face_gt_numpy = np.array(Image.open(face_gt_path))
            
            with torch.no_grad():
                face_xyxys = detect_face_crops([image_sub_numpy], yolov9_model, args.yolo_config, device)
                if face_xyxys:
                    H, W, _ = image_sub_numpy.shape
                    face_xyxys = [int(np.clip(face_xyxys[0], 0, W)), int(np.clip(face_xyxys[1], 0, H)), int(np.clip(face_xyxys[2], 0, W)), int(np.clip(face_xyxys[3], 0, H))]
                feat_face = inference(net_arcface, device, image_sub_numpy, face_xyxy=face_xyxys)
                feat_face_gt = inference(net_arcface, device, face_gt_numpy, face_xyxy=None)

                curr_feat_face = inference(face_cur_model, device, image_sub_numpy, face_xyxy=face_xyxys)
                curr_feat_face_gt = inference(face_cur_model, device, face_gt_numpy, face_xyxy=None)

            feat_face = feat_face / feat_face.norm(dim=-1, keepdim=True)
            feat_face_gt = feat_face_gt / feat_face_gt.norm(dim=-1, keepdim=True)
            face_sim = (feat_face @ feat_face_gt.T).item()
            tmp_face_sim.append(face_sim)

            curr_feat_face = curr_feat_face / curr_feat_face.norm(dim=-1, keepdim=True)
            curr_feat_face_gt = curr_feat_face_gt / curr_feat_face_gt.norm(dim=-1, keepdim=True)
            curr_face_sim = (curr_feat_face @ curr_feat_face_gt.T).item()
            tmp_curr_face_sim.append(curr_face_sim)
            
            sub_attr_clip_t = []
            sub_attr_clip_i = []
            sub_attr_dino_i = []
            for attr in attrs:
                with torch.no_grad():
                    det_results = owlv2_detect(model_owlv2, processor_owlv2, device, image_sub_numpy, attr)
                det_boxes, det_scores, det_text_labels = det_results["boxes"], det_results["scores"], det_results["text_labels"]

                if len(det_boxes) == 0:
                    image_sub_attr_numpy = image_sub_numpy
                    sub_attr_clip_t.append(0)
                    sub_attr_clip_i.append(0)
                    sub_attr_dino_i.append(0)
                    continue
                else:
                    box_index = torch.argmax(det_scores).item()
                    det_box = det_boxes[box_index].cpu().numpy().tolist()
                    det_box = [round(coord) for coord in det_box]
                    H, W, _ = image_sub_numpy.shape
                    det_box = [int(np.clip(det_box[0], 0, W)), int(np.clip(det_box[1], 0, H)), int(np.clip(det_box[2], 0, W)), int(np.clip(det_box[3], 0, H))]
                    image_sub_attr_numpy = image_sub_numpy[det_box[1]:det_box[3], det_box[0]:det_box[2]] if valid_bbox(det_box) else image_sub_numpy
                
                with torch.no_grad():
                    image_sub_attr = Image.fromarray(image_sub_attr_numpy)
                    inputs_image_attr = processor_clip(images=image_sub_attr, return_tensors="pt").to(device)
                    image_features_attr = model_clip.get_image_features(**inputs_image_attr)
                    image_features_attr = image_features_attr / image_features_attr.norm(dim=-1, keepdim=True)

                    inputs_image_attr_dinov2 = processor_dinov2(images=image_sub_attr, return_tensors="pt").to(device)
                    image_features_attr_dinov2 = model_dinov2(**inputs_image_attr_dinov2).pooler_output
                    image_features_attr_dinov2 = image_features_attr_dinov2 / image_features_attr_dinov2.norm(dim=-1, keepdim=True)

                    image_sub_attr_gt_path = os.path.join(item_path, "subjects", sub, attr + ".png")
                    image_sub_attr_gt = Image.open(image_sub_attr_gt_path)
                    inputs_image_attr_gt = processor_clip(images=image_sub_attr_gt, return_tensors="pt").to(device)
                    image_features_attr_gt = model_clip.get_image_features(**inputs_image_attr_gt)
                    image_features_attr_gt = image_features_attr_gt / image_features_attr_gt.norm(dim=-1, keepdim=True)

                    inputs_image_attr_gt_dinov2 = processor_dinov2(images=image_sub_attr_gt, return_tensors="pt").to(device)
                    image_features_attr_gt_dinov2 = model_dinov2(**inputs_image_attr_gt_dinov2).pooler_output
                    image_features_attr_gt_dinov2 = image_features_attr_gt_dinov2 / image_features_attr_gt_dinov2.norm(dim=-1, keepdim=True)

                    inputs_text_attr = tokenizer_clip(attr, padding=True, return_tensors="pt").to(device)
                    text_features_attr = model_clip.get_text_features(**inputs_text_attr)
                    text_features_attr = text_features_attr / text_features_attr.norm(dim=-1, keepdim=True)

                clip_t_attr = (image_features_attr @ text_features_attr.T).item()
                clip_i_attr = (image_features_attr @ image_features_attr_gt.T).item()
                dino_i_attr = (image_features_attr_dinov2 @ image_features_attr_gt_dinov2.T).item()
                sub_attr_clip_t.append(clip_t_attr)
                sub_attr_clip_i.append(clip_i_attr)
                sub_attr_dino_i.append(dino_i_attr)

            if len(sub_attr_clip_t):
                sub_clip_t.append(np.mean(sub_attr_clip_t))
            if len(sub_attr_clip_i):
                sub_clip_i.append(np.mean(sub_attr_clip_i))
            if len(sub_attr_dino_i):
                sub_dino_i.append(np.mean(sub_attr_dino_i))
        
        # 计算所有subjects的平均值
        if len(sub_clip_t) > 0:
            tmp_clip_t.append(np.mean(sub_clip_t))
        if len(sub_clip_i) > 0:
            tmp_clip_i.append(np.mean(sub_clip_i))
        if len(sub_dino_i) > 0:
            tmp_dino_i.append(np.mean(sub_dino_i))

        sub_object_clip_t = []
        sub_object_clip_i = []
        sub_object_dino_i = []

        if os.path.exists(os.path.join(item_path, "objects")):
            objects = ls(os.path.join(item_path, "objects"))
            for obj in objects:
                obj_path = os.path.join(item_path, "objects", obj)
                image_obj_gt = Image.open(obj_path)
                image_obj_gt_numpy = np.array(image_obj_gt)
                text_obj = obj.replace(".png", "")
                with torch.no_grad():
                    det_results_obj = owlv2_detect(model_owlv2, processor_owlv2, device, image_numpy, text_obj)
                det_boxes_obj, det_scores_obj, det_text_labels_obj = det_results_obj["boxes"], det_results_obj["scores"], det_results_obj["text_labels"]
                if len(det_boxes_obj) == 0:
                    image_obj_numpy = image_numpy
                    sub_object_clip_t.append(0)
                    sub_object_clip_i.append(0)
                    sub_object_dino_i.append(0)
                    continue
                else:
                    box_index_obj = torch.argmax(det_scores_obj).item()
                    det_box_obj = det_boxes_obj[box_index_obj].cpu().numpy().tolist()
                    det_box_obj = [round(coord) for coord in det_box_obj]
                    H, W, _ = image_numpy.shape
                    det_box_obj = [int(np.clip(det_box_obj[0], 0, W)), int(np.clip(det_box_obj[1], 0, H)), int(np.clip(det_box_obj[2], 0, W)), int(np.clip(det_box_obj[3], 0, H))]
                    image_obj_numpy = image_numpy[det_box_obj[1]:det_box_obj[3], det_box_obj[0]:det_box_obj[2]] if valid_bbox(det_box_obj) else image_numpy
                with torch.no_grad():
                    image_obj = Image.fromarray(image_obj_numpy)
                    inputs_image_obj = processor_clip(images=image_obj, return_tensors="pt").to(device)
                    image_features_obj = model_clip.get_image_features(**inputs_image_obj)
                    image_features_obj = image_features_obj / image_features_obj.norm(dim=-1, keepdim=True)

                    inputs_image_obj_dinov2 = processor_dinov2(images=image_obj, return_tensors="pt").to(device)
                    image_features_obj_dinov2 = model_dinov2(**inputs_image_obj_dinov2).pooler_output
                    image_features_obj_dinov2 = image_features_obj_dinov2 / image_features_obj_dinov2.norm(dim=-1, keepdim=True)

                    inputs_image_obj_gt = processor_clip(images=image_obj_gt, return_tensors="pt").to(device)
                    image_features_obj_gt = model_clip.get_image_features(**inputs_image_obj_gt)
                    image_features_obj_gt = image_features_obj_gt / image_features_obj_gt.norm(dim=-1, keepdim=True)

                    inputs_image_obj_gt_dinov2 = processor_dinov2(images=image_obj_gt, return_tensors="pt").to(device)
                    image_features_obj_gt_dinov2 = model_dinov2(**inputs_image_obj_gt_dinov2).pooler_output
                    image_features_obj_gt_dinov2 = image_features_obj_gt_dinov2 / image_features_obj_gt_dinov2.norm(dim=-1, keepdim=True)

                    inputs_text_obj = tokenizer_clip(text_obj, padding=True, return_tensors="pt").to(device)
                    text_features_obj = model_clip.get_text_features(**inputs_text_obj)
                    text_features_obj = text_features_obj / text_features_obj.norm(dim=-1, keepdim=True)

                clip_t_obj = (image_features_obj @ text_features_obj.T).item()
                clip_i_obj = (image_features_obj @ image_features_obj_gt.T).item()
                dino_i_obj = (image_features_obj_dinov2 @ image_features_obj_gt_dinov2.T).item()
                sub_object_clip_t.append(clip_t_obj)
                sub_object_clip_i.append(clip_i_obj)
                sub_object_dino_i.append(dino_i_obj)
            if len(sub_object_clip_t) > 0:
                tmp_clip_t.append(np.mean(sub_object_clip_t))
                tmp_clip_i.append(np.mean(sub_object_clip_i))
                tmp_dino_i.append(np.mean(sub_object_dino_i))
        
        # 收集当前视频的结果到对应的person组和overall组
        # 确保每个视频都有所有指标的值，如果没有有效值则使用0.0
        face_sim_mean = np.mean(tmp_face_sim) if len(tmp_face_sim) > 0 else 0.0
        curr_face_sim_mean = np.mean(tmp_curr_face_sim) if len(tmp_curr_face_sim) > 0 else 0.0
        clip_t_mean = np.mean(tmp_clip_t) if len(tmp_clip_t) > 0 else 0.0
        clip_i_mean = np.mean(tmp_clip_i) if len(tmp_clip_i) > 0 else 0.0
        dino_i_mean = np.mean(tmp_dino_i) if len(tmp_dino_i) > 0 else 0.0
        
        # 为当前视频添加所有指标的值（每个视频都有）
        res_by_person[person_key]['face_sim'].append(face_sim_mean)
        res_by_person[person_key]['curr_face_sim'].append(curr_face_sim_mean)
        res_by_person[person_key]['clip_t'].append(clip_t_mean)
        res_by_person[person_key]['clip_i'].append(clip_i_mean)
        res_by_person[person_key]['dino_i'].append(dino_i_mean)
        
        res_by_person['overall']['face_sim'].append(face_sim_mean)
        res_by_person['overall']['curr_face_sim'].append(curr_face_sim_mean)
        res_by_person['overall']['clip_t'].append(clip_t_mean)
        res_by_person['overall']['clip_i'].append(clip_i_mean)
        res_by_person['overall']['dino_i'].append(dino_i_mean)

    # 计算每个person和整体的指标
    overall_final_results = None
    for person_name in ['person1', 'person2', 'person3', 'overall']:
        person_data = res_by_person[person_name]
        
        # 初始化结果（使用float，因为现在存储的是标量）
        clip_t_sum = 0.0
        clip_i_sum = 0.0
        dino_i_sum = 0.0
        face_sim_sum = 0.0
        curr_face_sim_sum = 0.0
        video_count = 0
        
        if len(person_data['clip_t']) > 0:
            clip_t_sum = sum(person_data['clip_t'])
            clip_i_sum = sum(person_data['clip_i'])
            dino_i_sum = sum(person_data['dino_i'])
            video_count = len(person_data['clip_t'])
        
        if len(person_data['face_sim']) > 0:
            face_sim_sum = sum(person_data['face_sim'])
            curr_face_sim_sum = sum(person_data['curr_face_sim'])
        
        # 转换为张量用于跨进程聚合
        clip_t_sum_tensor = torch.tensor(clip_t_sum, dtype=torch.float32, device=device)
        clip_i_sum_tensor = torch.tensor(clip_i_sum, dtype=torch.float32, device=device)
        dino_i_sum_tensor = torch.tensor(dino_i_sum, dtype=torch.float32, device=device)
        face_sim_sum_tensor = torch.tensor(face_sim_sum, dtype=torch.float32, device=device)
        curr_face_sim_sum_tensor = torch.tensor(curr_face_sim_sum, dtype=torch.float32, device=device)
        video_count_tensor = torch.tensor(video_count, dtype=torch.int64, device=device)
        
        # 跨进程聚合
        dist.all_reduce(clip_t_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(clip_i_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(dino_i_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(face_sim_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(curr_face_sim_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(video_count_tensor, op=dist.ReduceOp.SUM)
        
        # 计算平均值
        if video_count_tensor.item() > 0:
            clip_t_avg = (clip_t_sum_tensor / video_count_tensor).item()
            clip_i_avg = (clip_i_sum_tensor / video_count_tensor).item()
            dino_i_avg = (dino_i_sum_tensor / video_count_tensor).item()
            face_sim_avg = (face_sim_sum_tensor / video_count_tensor).item()
            curr_face_sim_avg = (curr_face_sim_sum_tensor / video_count_tensor).item()
        else:
            clip_t_avg = 0.0
            clip_i_avg = 0.0
            dino_i_avg = 0.0
            face_sim_avg = 0.0
            curr_face_sim_avg = 0.0
        
        # 只在主进程打印结果，并保存overall的结果用于最终总结
        if is_main_process:
            log_print(f'\n{"="*60}')
            log_print(f'{person_name.upper()} Results (Total Videos: {video_count_tensor.item()})')
            log_print(f'{"="*60}')
            log_print(f'CLIP_T: {clip_t_avg:.4f}')
            log_print(f'CLIP_I: {clip_i_avg:.4f}')
            log_print(f'DINO_I: {dino_i_avg:.4f}')
            log_print(f'FaceSim: {face_sim_avg:.4f}')
            log_print(f'FaceCurSim: {curr_face_sim_avg:.4f}')
            
            # 保存overall的结果用于最终总结
            if person_name == 'overall':
                overall_final_results = {
                    'clip_t': clip_t_avg,
                    'clip_i': clip_i_avg,
                    'dino_i': dino_i_avg,
                    'face_sim': face_sim_avg,
                    'curr_face_sim': curr_face_sim_avg,
                    'video_count': video_count_tensor.item()
                }

    # 在主进程输出最终总结
    if is_main_process and overall_final_results is not None:
        log_print(f'\n{"="*60}')
        log_print(f'FINAL SUMMARY - Overall Results')
        log_print(f'{"="*60}')
        log_print(f'Total Videos Evaluated: {overall_final_results["video_count"]}')
        log_print(f'CLIP_T: {overall_final_results["clip_t"]:.4f}')
        log_print(f'CLIP_I: {overall_final_results["clip_i"]:.4f}')
        log_print(f'DINO_I: {overall_final_results["dino_i"]:.4f}')
        log_print(f'FaceSim: {overall_final_results["face_sim"]:.4f}')
        log_print(f'FaceCurSim: {overall_final_results["curr_face_sim"]:.4f}')
        log_print(f'{"="*60}')
    
    # 关闭log文件（仅在主进程）
    if is_main_process and log_file is not None:
        log_file.close()
        print(f"Log file saved to: ./results/{log_name}.log")
    
    # 清理分布式环境
    if dist.is_initialized():
        dist.destroy_process_group()