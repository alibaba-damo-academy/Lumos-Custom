import os
import sys
import argparse
import glob
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.distributed as dist

# Make sure VideoCLIP_XL code under benchmark/models is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models", "VideoCLIP_XL"))
from modeling import VideoCLIP_XL
from utils.text_encoder import text_encoder
from fs_oss import read, ls

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
        description="Distributed evaluation for global video-text and video-video consistency using VideoCLIP_XL"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="The path to the generated video directory.")
    parser.add_argument(
        "--ref_image",
        type=str,
        required=True,
        help="The path to the reference image directory.")
    parser.add_argument(
        "--videoclip_model_path",
        type=str,
        required=True,
        help="Path to VideoCLIP_XL model weights (.bin file)")
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
        os.makedirs("./results_global", exist_ok=True)
        log_file_path = os.path.join("./results_global", f"{log_name}.log")
        log_file = open(log_file_path, 'w', encoding='utf-8')
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Global Evaluation Results for: {log_name}\n")
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Video Directory: {args.video_dir}\n")
        log_file.write(f"Reference Image Directory: {args.ref_image}\n")
        log_file.write(f"VideoCLIP Model Path: {args.videoclip_model_path}\n")
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

    device = "cuda"
    videoclip_xl = VideoCLIP_XL()
    state_dict = torch.load(args.videoclip_model_path, map_location="cpu")
    videoclip_xl.load_state_dict(state_dict)
    videoclip_xl.eval().to(device)

    def _frame_from_video(video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break  
            
    v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    def normalize(data):
        return (data / 255.0 - v_mean) / v_std
            
    def video_preprocessing(video_path, start_frame=None, end_time=None, fnum=8):
        if video_path.startswith('oss'):
            import tempfile
            video_path = read(video_path)
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_file:
                temp_file.write(video_path.getbuffer())
                temp_file.flush()
                video = cv2.VideoCapture(temp_file.name)
        else:
            video = cv2.VideoCapture(video_path)
        
        frames = [x for x in _frame_from_video(video)]
        if start_frame and end_time:
            fps = int(video.get(cv2.CAP_PROP_FPS))
            end = fps * end_time
            start = start_frame
            frames = frames[start: end]
        step = len(frames) // fnum
        frames = frames[::step][:fnum]
        vid_tube = []
        for fr in frames:
            fr = fr[:,:,::-1]
            fr = cv2.resize(fr, (224, 224))
            fr = np.expand_dims(normalize(fr), axis=(0, 1))
            vid_tube.append(fr) 
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube)
        return vid_tube

    test_path = args.ref_image
    test_video_dir = args.video_dir
    files = ls(test_path)

    if len(files) % num_processes != 0:
        files = files + files[:(num_processes - len(files) % num_processes)]
    files = files[local_rank::num_processes]

    res_by_person = {
        'person1': {'clip_t': [], 'clip_v': []},
        'person2': {'clip_t': [], 'clip_v': []},
        'person3': {'clip_t': [], 'clip_v': []},
        'overall': {'clip_t': [], 'clip_v': []}
    }
    
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

    for item in tqdm(files, disable=not is_main_process):
        video_name = f"{item}*.mp4"
        item_path = os.path.join(test_path, item)
        caption_data = os.path.join(item_path, "caption.txt")
        caption = open(caption_data, 'r', encoding='utf-8').read().split(";", 1)[1]
        video_ori = os.path.join(item_path, "video.mp4")
        video_generate = glob.glob(os.path.join(test_video_dir, video_name))[0]
        video_basename = os.path.basename(video_generate)
        person_key = extract_person_from_video_name(video_basename)
        
        texts = [caption]
        with torch.no_grad():
            video_inputs = torch.cat([video_preprocessing(video_ori, start_frame=10, end_time=5), video_preprocessing(video_generate)], 0).float().to(device)
            video_features = videoclip_xl.vision_model.get_vid_features(video_inputs).float()
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

            text_inputs = text_encoder.tokenize(texts, truncate=True).to(device)
            text_features = videoclip_xl.text_model.encode_text(text_inputs).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        clip_t = (text_features @ video_features[1].T).item()
        clip_v = (video_features[0] @ video_features[1].T).item()
        
        res_by_person[person_key]['clip_t'].append(clip_t)
        res_by_person[person_key]['clip_v'].append(clip_v)
        res_by_person['overall']['clip_t'].append(clip_t)
        res_by_person['overall']['clip_v'].append(clip_v)

        dist.barrier()

    overall_final_results = None
    for person_name in ['person1', 'person2', 'person3', 'overall']:
        person_data = res_by_person[person_name]
        
        clip_t_sum = 0.0
        clip_v_sum = 0.0
        video_count = 0
        
        if len(person_data['clip_t']) > 0:
            clip_t_sum = sum(person_data['clip_t'])
            clip_v_sum = sum(person_data['clip_v'])
            video_count = len(person_data['clip_t'])
        
        clip_t_sum_tensor = torch.tensor(clip_t_sum, dtype=torch.float32, device=device)
        clip_v_sum_tensor = torch.tensor(clip_v_sum, dtype=torch.float32, device=device)
        video_count_tensor = torch.tensor(video_count, dtype=torch.int64, device=device)
        
        dist.all_reduce(clip_t_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(clip_v_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(video_count_tensor, op=dist.ReduceOp.SUM)
        
        if video_count_tensor.item() > 0:
            clip_t_avg = (clip_t_sum_tensor / video_count_tensor).item()
            clip_v_avg = (clip_v_sum_tensor / video_count_tensor).item()
        else:
            clip_t_avg = 0.0
            clip_v_avg = 0.0
        
        if is_main_process:
            log_print(f'\n{"="*60}')
            log_print(f'{person_name.upper()} Results (Total Videos: {video_count_tensor.item()})')
            log_print(f'{"="*60}')
            log_print(f'CLIP_T: {clip_t_avg:.4f}')
            log_print(f'CLIP_V: {clip_v_avg:.4f}')
            
            if person_name == 'overall':
                overall_final_results = {
                    'clip_t': clip_t_avg,
                    'clip_v': clip_v_avg,
                    'video_count': video_count_tensor.item()
                }

    if is_main_process and overall_final_results is not None:
        log_print(f'\n{"="*60}')
        log_print(f'FINAL SUMMARY - Overall Results')
        log_print(f'{"="*60}')
        log_print(f'Total Videos Evaluated: {overall_final_results["video_count"]}')
        log_print(f'CLIP_T: {overall_final_results["clip_t"]:.4f}')
        log_print(f'CLIP_V: {overall_final_results["clip_v"]:.4f}')
        log_print(f'{"="*60}')
    
    if is_main_process and log_file is not None:
        log_file.close()
        print(f"Log file saved to: ./results_global/{log_name}.log")
    
    if dist.is_initialized():
        dist.destroy_process_group()        