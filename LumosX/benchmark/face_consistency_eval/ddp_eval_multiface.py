import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import argparse
import cv2
import decord
decord.bridge.set_bridge("torch")
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    from numpy.core.multiarray import _reconstruct
    torch.serialization.add_safe_globals([_reconstruct])
except ImportError:
    pass

from distributed_arcface_similarity_multiface import DistributedArcfaceSimilarity

def init_dist(launcher="pytorch", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)    
    return local_rank

@torch.no_grad()
def extract_arcface_embeddings(images, model):
    def load_image(pil_image):
        img = np.array(pil_image)
        img = cv2.resize(img, (112, 112))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        img.div_(255).sub_(0.5).div_(0.5)
        return img

    images = [load_image(img) for img in images]
    images = torch.stack(images)
    images = images.to("cuda")
    features = model(images).cpu()
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed evaluation for multiface similarity using ArcFace and CurricularFace"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="The path to the video directory.")
    parser.add_argument(
        "--ref_image",
        type=str,
        required=True,
        help="The path to the reference image directory.")
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="../models/yolov9/weight/best.pt",
        help="Path to YOLOv9 weights")
    parser.add_argument(
        "--yolo_config",
        type=str,
        default="../models/yolov9/data/coco.yaml",
        help="Path to YOLOv9 config")
    parser.add_argument(
        "--arcface_weights",
        type=str,
        default="../models/arcface/weight/backbone.pth",
        help="Path to ArcFace weights")
    parser.add_argument(
        "--curricular_face_weights",
        type=str,
        required=True,
        help="Path to CurricularFace weights")    
    args = parser.parse_args()

    local_rank      = init_dist()
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    log_file = None
    log_name = None
    if is_main_process:
        video_dir_parts = args.video_dir.rstrip('/').split('/')
        log_name = video_dir_parts[-1] if video_dir_parts[-1] else video_dir_parts[-2]
        os.makedirs("./results_multiface", exist_ok=True)
        log_file_path = os.path.join("./results_multiface", f"{log_name}.log")
        log_file = open(log_file_path, 'w', encoding='utf-8')
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Multiface Evaluation Results for: {log_name}\n")
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Video Directory: {args.video_dir}\n")
        log_file.write(f"Reference Image Directory: {args.ref_image}\n")
        log_file.write(f"Total Processes: {num_processes}\n")
        log_file.write(f"{'='*60}\n\n")
        log_file.flush()
    
    def log_print(*args, **kwargs):
        if is_main_process:
            end = kwargs.get('end', '\n')
            print(*args, **kwargs)
            if log_file is not None:
                output = ' '.join(str(arg) for arg in args)
                log_file.write(output + end)
                log_file.flush()

    distributed_similarity_models = {
        'person1': DistributedArcfaceSimilarity(
            yolov9_config={
                "weights": args.yolo_weights,
                "class_data": args.yolo_config
            },
            arcface_config={
                "network": "r100",
                "weights": args.arcface_weights,
                "curricular_face_weights": args.curricular_face_weights
            }
        ),
        'person2': DistributedArcfaceSimilarity(
            yolov9_config={
                "weights": args.yolo_weights,
                "class_data": args.yolo_config
            },
            arcface_config={
                "network": "r100",
                "weights": args.arcface_weights,
                "curricular_face_weights": args.curricular_face_weights
            }
        ),
        'person3': DistributedArcfaceSimilarity(
            yolov9_config={
                "weights": args.yolo_weights,
                "class_data": args.yolo_config
            },
            arcface_config={
                "network": "r100",
                "weights": args.arcface_weights,
                "curricular_face_weights": args.curricular_face_weights
            }
        ),
        'overall': DistributedArcfaceSimilarity(
            yolov9_config={
                "weights": args.yolo_weights,
                "class_data": args.yolo_config
            },
            arcface_config={
                "network": "r100",
                "weights": args.arcface_weights,
                "curricular_face_weights": args.curricular_face_weights
            }
        )
    }

    test_dirs = os.listdir(args.ref_image)
    if len(test_dirs) % num_processes != 0:
        test_dirs = test_dirs + test_dirs[:(num_processes - len(test_dirs) % num_processes)]
    test_dirs = test_dirs[local_rank::num_processes]

    def extract_person_from_video_name(video_name):
        video_name_lower = video_name.lower()
        if 'person1' in video_name_lower or 'video_person1' in video_name_lower:
            return 'person1'
        elif 'person2' in video_name_lower or 'video_person2' in video_name_lower:
            return 'person2'
        elif 'person3' in video_name_lower or 'video_person3' in video_name_lower:
            return 'person3'
        else:
            return 'overall'

    for test_dir in tqdm(test_dirs, disable=not is_main_process):
        path = os.path.join(args.ref_image, test_dir)
        video_dir = glob.glob(os.path.join(args.video_dir, test_dir+"*.mp4"))[0]
        video_basename = os.path.basename(video_dir)
        person_key = extract_person_from_video_name(video_basename)
        
        face_images = glob.glob(os.path.join(path, "subjects/*/face*.png"))
        face_images = [Image.open(image).convert("RGB") for image in face_images]

        video_reader = decord.VideoReader(video_dir)
        videos = video_reader.get_batch(np.arange(len(video_reader)))

        face_embeddings = extract_arcface_embeddings(face_images, distributed_similarity_models[person_key].arceface_model)
        face_embeddings_cur = extract_arcface_embeddings(face_images, distributed_similarity_models[person_key].face_cur_model)
        
        distributed_similarity_models[person_key].accumulate_stats(videos[None], face_embeddings[None], face_embeddings_cur[None])
        distributed_similarity_models['overall'].accumulate_stats(videos[None], face_embeddings[None], face_embeddings_cur[None])

        dist.barrier()

    overall_final_results = None
    for person_name in ['person1', 'person2', 'person3', 'overall']:
        similarity, similarity_sum_cur = distributed_similarity_models[person_name].get_statistics()
        
        if is_main_process:
            similarity_value = similarity.item() if hasattr(similarity, 'item') else similarity
            similarity_sum_cur_value = similarity_sum_cur.item() if hasattr(similarity_sum_cur, 'item') else similarity_sum_cur
            
            log_print(f'\n{"="*60}')
            log_print(f'{person_name.upper()} Results')
            log_print(f'{"="*60}')
            log_print(f'Arcface Similarity: {similarity_value:.4f}')
            log_print(f'Similarity Sum Cur: {similarity_sum_cur_value:.4f}')
            
            if person_name == 'overall':
                overall_final_results = {
                    'arcface_similarity': similarity_value,
                    'similarity_sum_cur': similarity_sum_cur_value
                }

    if is_main_process and overall_final_results is not None:
        log_print(f'\n{"="*60}')
        log_print(f'FINAL SUMMARY - Overall Results')
        log_print(f'{"="*60}')
        log_print(f'Arcface Similarity: {overall_final_results["arcface_similarity"]:.4f}')
        log_print(f'Similarity Sum Cur: {overall_final_results["similarity_sum_cur"]:.4f}')
        log_print(f'{"="*60}')
    
    if is_main_process and log_file is not None:
        log_file.close()
        print(f"Log file saved to: ./results_multiface/{log_name}.log")
    
    if dist.is_initialized():
        dist.destroy_process_group()