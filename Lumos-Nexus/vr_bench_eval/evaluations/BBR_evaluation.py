#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Evaluator for Biological Behavior Reasoning (BBR)
"""

import os
import json
import time
from datetime import datetime
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluators.BBR_evaluator import BiologicalBehaviorEvaluator
from vlm_utils import prepare_vlm_path

VIDEO_ROOT = os.environ.get("VR_BENCH_VIDEO_ROOT")
OUTPUT_ROOT = os.path.join(_ROOT, "eval_results", "eval_results_BBR_v1")
PROMPT_FILE = os.path.join(_ROOT, "prompts_checked", "BBR_prompts.json")
_VLM_DEFAULT = os.path.join(os.path.dirname(_ROOT), "models", "Qwen3-VL-30B-A3B-Instruct")
VLM_PATH = os.environ.get("VR_BENCH_VLM_PATH", _VLM_DEFAULT)

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def load_prompt_data(prompt_json_path):
    """Load BBR prompt definitions from JSON."""
    with open(prompt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts_data = data["Biological Behavior Reasoning"]["prompts"]
    prompt_map = {p["id"]: p for p in prompts_data}
    print(f"📚 Loaded {len(prompt_map)} Biological Behavior Reasoning prompts.")
    return prompt_map


def evaluate_all_BBR_videos(video_root, model_subdir, prompt_map, evaluator):
    """Batch-evaluate BBR videos for one model checkpoint."""
    subdir_path = os.path.join(video_root, model_subdir, "biological_behavior_reasoning")
    if not os.path.exists(subdir_path):
        print(f"⚠️ Folder not found: {subdir_path}")
        return []

    print(f"\n🚀 Evaluating Biological Behavior Reasoning for model: {model_subdir}")
    results_summary = []

    for filename in os.listdir(subdir_path):
        if not filename.endswith(".mp4"):
            continue

        vid_id = "_".join(filename.split("_")[:2]) if "_" in filename else filename.split(".")[0]
        vid_id = vid_id[:-4]

        if vid_id not in prompt_map:
            print(f"⚠️ No prompt found for {vid_id}, skipping...")
            continue

        json_data = prompt_map[vid_id]
        video_path = os.path.join(subdir_path, filename)

        print(f"\n🎬 Processing {filename} ({vid_id})")
        print(f"🧩 Prompt: {json_data['prompt_en']}")
        print(f"📝 Description: {json_data['description']}")

        try:
            start = time.perf_counter()
            result = evaluator.evaluate_video(json_data, video_path)
            end = time.perf_counter()
            elapsed = round(end - start, 2)

            save_dir = os.path.join(OUTPUT_ROOT, model_subdir, "biological_behavior_reasoning")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{vid_id}.json")

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            score_result = result["scoring_results"]
            layer_scores = score_result["reasoning_layer_scores"]

            results_summary.append({
                "video_id": vid_id,
                "overall_score": score_result["weighted_average_score"],
                "biomechanical_realism_score": layer_scores["biomechanical_realism"]["score"],
                "environmental_interaction_score": layer_scores["environmental_interaction"]["score"],
                "ecological_coherence_score": layer_scores["ecological_coherence"]["score"],
                "time_seconds": elapsed,
                "file_path": save_path
            })

            print(f"✅ Saved → {save_path}")
            print(f"📊 Overall Score: {score_result['weighted_average_score']}")
            print(f"⏱️ Time used: {elapsed:.2f}s\n")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if results_summary:
        summary_path = os.path.join(OUTPUT_ROOT, model_subdir, "BBR_evaluation_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        overall = [x["overall_score"] for x in results_summary]
        s1 = [x["biomechanical_realism_score"] for x in results_summary]
        s2 = [x["environmental_interaction_score"] for x in results_summary]
        s3 = [x["ecological_coherence_score"] for x in results_summary]

        summary = {
            "model_name": model_subdir,
            "total_videos": len(results_summary),
            "average_scores": {
                "overall": round(sum(overall) / len(overall), 4),
                "biomechanical_realism": round(sum(s1) / len(s1), 4),
                "environmental_interaction": round(sum(s2) / len(s2), 4),
                "ecological_coherence": round(sum(s3) / len(s3), 4)
            },
            "results": results_summary,
            "timestamp": datetime.now().isoformat()
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"📋 Summary saved: {summary_path}")

    return results_summary


if __name__ == "__main__":
    prompt_map = load_prompt_data(PROMPT_FILE)
    VLM_PATH = prepare_vlm_path(VLM_PATH)
    evaluator = BiologicalBehaviorEvaluator(vlm_path=VLM_PATH, save_dir=OUTPUT_ROOT, seed=42, deterministic=True)

    if not VIDEO_ROOT:
        raise ValueError("VR_BENCH_VIDEO_ROOT is required. Please set it in run_qwen.sh or export it before running.")

    models_env = os.environ.get("VR_BENCH_MODELS", "")
    MODELS = [m.strip() for m in models_env.split(",") if m.strip()]
    if not MODELS:
        raise ValueError("VR_BENCH_MODELS is required. Please set it in run_qwen.sh or export it before running.")
    all_results = {}

    for model in MODELS:
        print(f"\n{'='*60}\n🏁 Evaluating {model}\n{'='*60}")
        results = evaluate_all_BBR_videos(VIDEO_ROOT, model, prompt_map, evaluator)
        all_results[model] = results

    overall_summary = {
        "evaluation_date": datetime.now().isoformat(),
        "evaluation_type": "Biological Behavior Reasoning",
        "total_models": len(MODELS),
        "models_evaluated": MODELS,
        "model_performance": {}
    }

    for model, results in all_results.items():
        if not results:
            continue
        overall = [x["overall_score"] for x in results]
        s1 = [x["biomechanical_realism_score"] for x in results]
        s2 = [x["environmental_interaction_score"] for x in results]
        s3 = [x["ecological_coherence_score"] for x in results]
        overall_summary["model_performance"][model] = {
            "average_overall_score": round(sum(overall) / len(overall), 4),
            "average_biomechanical_realism": round(sum(s1) / len(s1), 4),
            "average_environmental_interaction": round(sum(s2) / len(s2), 4),
            "average_ecological_coherence": round(sum(s3) / len(s3), 4),
            "total_videos": len(results),
            "min_overall_score": round(min(overall), 4),
            "max_overall_score": round(max(overall), 4)
        }

    overall_path = os.path.join(OUTPUT_ROOT, "BBR_overall_evaluation_report.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print(f"\n🎯 All Biological Behavior Reasoning evaluations completed!")
    print(f"📊 Overall report saved: {overall_path}\n")

    for model, perf in overall_summary["model_performance"].items():
        print(f"  {model}:")
        print(f"    Overall: {perf['average_overall_score']}")
        print(f"    Biomechanical Realism: {perf['average_biomechanical_realism']}")
        print(f"    Environmental Interaction: {perf['average_environmental_interaction']}")
        print(f"    Ecological Coherence: {perf['average_ecological_coherence']}")
        print(f"    Videos: {perf['total_videos']}")
