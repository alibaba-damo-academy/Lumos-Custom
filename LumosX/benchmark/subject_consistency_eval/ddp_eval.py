import os
import glob
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
import decord
import argparse
from tqdm import tqdm
import warnings
from easydict import EasyDict as edict
from dynamic_degree import DistributedDynamicDegree

warnings.filterwarnings("ignore")
decord.bridge.set_bridge("torch")


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
        description="Video quality evaluation script for dynamic degree metrics"
    )
    parser.add_argument("--video_dir", type=str, required=True, help="Path to generated videos directory")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to reference images directory")
    parser.add_argument("--raft_model_path", type=str, default="../models/raft/models/raft-things.pth", help="Path to RAFT model weights")    
    args = parser.parse_args()

    local_rank      = init_dist()
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    # 设置log文件（仅在主进程）
    log_file = None
    log_name = None
    if is_main_process:
        # 从video_dir中提取最后一部分作为文件名
        video_dir_parts = args.video_dir.rstrip('/').split('/')
        log_name = video_dir_parts[-1] if video_dir_parts[-1] else video_dir_parts[-2]
        
        # 创建results目录
        os.makedirs("./results_quality", exist_ok=True)
        
        # 创建log文件
        log_file_path = os.path.join("./results_quality", f"{log_name}.log")
        log_file = open(log_file_path, 'w', encoding='utf-8')
        
        # 写入log抬头
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Quality Evaluation Results for: {log_name}\n")
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

    dynamic_degree_models = {
        'person1': DistributedDynamicDegree(edict({
            "model": args.raft_model_path,
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        })),
        'person2': DistributedDynamicDegree(edict({
            "model": args.raft_model_path,
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        })),
        'person3': DistributedDynamicDegree(edict({
            "model": args.raft_model_path,
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        })),
        'overall': DistributedDynamicDegree(edict({
            "model": args.raft_model_path,
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        }))
    }

    test_dirs = os.listdir(args.ref_image)
    if len(test_dirs) % num_processes != 0:
        test_dirs = test_dirs + test_dirs[:(num_processes - len(test_dirs) % num_processes)]
    test_dirs = test_dirs[local_rank::num_processes]

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

    for test_dir in tqdm(test_dirs, disable=not is_main_process):
        path = os.path.join(args.ref_image, test_dir)
        video_dir = glob.glob(os.path.join(args.video_dir, test_dir+"*.mp4"))[0]
        # 从视频路径中提取person信息
        video_basename = os.path.basename(video_dir)
        person_key = extract_person_from_video_name(video_basename)
        
        face_image = glob.glob(os.path.join(path, "subjects/1/face*.png"))[0]
        face_image = Image.open(face_image).convert("RGB")

        video_reader = decord.VideoReader(video_dir)
        fps = video_reader.get_avg_fps()
        
        videos = video_reader.get_batch(np.arange(len(video_reader)))[5:]
        videos = videos.permute(0, 3, 1, 2).contiguous()
        dynamic_degree_models[person_key].accumulate_stats(videos, fps)
        dynamic_degree_models['overall'].accumulate_stats(videos, fps)

        dist.barrier()

    overall_final_results = None
    for person_name in ['person1', 'person2', 'person3', 'overall']:
        dynamic = dynamic_degree_models[person_name].get_statistics()
        
        if is_main_process:
            dynamic_value = dynamic.item() if hasattr(dynamic, 'item') else dynamic
            log_print(f'\n{"="*60}')
            log_print(f'{person_name.upper()} Results')
            log_print(f'{"="*60}')
            log_print(f'Dynamic: {dynamic_value:.4f}')
            
            # 保存overall的结果用于最终总结
            if person_name == 'overall':
                overall_final_results = {
                    'dynamic': dynamic_value
                }

    # 在主进程输出最终总结
    if is_main_process and overall_final_results is not None:
        log_print(f'\n{"="*60}')
        log_print(f'FINAL SUMMARY - Overall Results')
        log_print(f'{"="*60}')
        log_print(f'Dynamic: {overall_final_results["dynamic"]:.4f}')
        log_print(f'{"="*60}')
    
    # 关闭log文件（仅在主进程）
    if is_main_process and log_file is not None:
        log_file.close()
        print(f"Log file saved to: ./results_quality/{log_name}.log")
    
    if dist.is_initialized():
        dist.destroy_process_group()