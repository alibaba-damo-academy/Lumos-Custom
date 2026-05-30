#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluator for Cultural Commonsense Reasoning (CCR)
"""

import os
import re
import json
import torch
import numpy as np
from typing import Dict, List
from datetime import datetime
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
from decord import VideoReader, cpu


class CulturalReasoningEvaluator:
    """
    Evaluator for Cultural Commonsense Reasoning (CCR):
    1. Reads structured evaluation JSON with multiple questions.
    2. Builds multimodal prompt (video + text-based checklist).
    3. Generates model reasoning output and parses into structured JSON.
    4. Computes weighted average and reasoning-layer scores:
         - q1–2 → cultural_symbol_understanding
         - q3–5 → contextual_alignment
         - q6–8 → social_behavior_understanding
    """

    def __init__(
        self,
        vlm_path: str = "/mnt/workspace/cailingling/models/Qwen/Qwen3-VL-30B-A3B-Instruct",
        save_dir: str = "./evaluation_results/ccr_v1",
        seed: int = 42,
        deterministic: bool = True
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = seed
        self.deterministic = deterministic

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            print(f"🔒 Deterministic mode enabled (seed={seed})")


        print("🔹 Loading video-language model...")
        self.vlm = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            vlm_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(vlm_path)
        print("✅ VLM loaded successfully.")

        self.save_dir = os.path.expanduser(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

    def extract_frames(self, video_path: str, num_frames: int = 16) -> List[np.ndarray]:
        """Extract uniformly sampled frames from the video."""
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
        return [vr[i].asnumpy() for i in indices]

    def build_prompt_from_json_data(self, json_data: Dict) -> str:
        """Convert structured evaluation JSON into multimodal prompt."""
        pid = json_data["id"]
        category = json_data.get("category", "Cultural Commonsense Reasoning")
        prompt = json_data["prompt_en"]
        description = json_data.get("description", "")
        eval_info = json_data.get("evaluation", {})
        questions = eval_info.get("evaluation_questions", [])
        eval_description = eval_info.get("evaluation_description", "")

        q_text = "\n".join([f"{i+1}. {q['question']} (weight: {q['weight']})" for i, q in enumerate(questions)])
        instruction = eval_info.get("overall_instruction", "Answer 'Yes' or 'No' for each question and justify briefly.")
        answers_structure = ",\n    ".join(
            [f'"{q["id"]}": {{"answer": "Yes or No", "reason": "..."}}' for q in questions]
        )

        prompt_text = f"""
You are an objective cultural evaluator analyzing videos for cultural commonsense reasoning.
Your goal is to assess whether the video accurately represents cultural symbols,
attire, rituals, and social interactions consistent with the described culture.

EVALUATION PRINCIPLES:
1. Answer "Yes" if you can observe the described behavior/property in the video frames, even if it's partially visible or somewhat ambiguous
2. Answer "Yes" if the behavior/property is reasonably demonstrated, even if not perfect
3. Give the benefit of the doubt - if the behavior/property appears to be present or can be reasonably inferred, answer "Yes"
4. Answer "No" only if the behavior/property is clearly absent or contradicts what is shown
5. Consider the overall context and intent of the prompt when evaluating
6. Each question should be evaluated based on what you observe in the video frames

=== Video Task ===
Video ID: {pid}
Prompt: {prompt}
Description: {description}

=== Evaluation Context ===
{eval_description}

=== Evaluation Questions ===
{q_text}

{instruction}

=== Required JSON Output ===
{{
  "video_id": "{pid}",
  "answers": {{
    {answers_structure}
  }}
}}

IMPORTANT RULES:
- Output only valid JSON. Each answer must be exactly "Yes" or "No" (case-sensitive)
- Be reasonable: Answer "Yes" when the behavior/property is present or can be reasonably inferred
- If the behavior/property appears to be present (even partially), answer "Yes"
- Only answer "No" if the behavior/property is clearly absent
- Provide brief reasoning in the "reason" field explaining your decision
"""
        return prompt_text.strip()

    def ask_vlm(self, video_frames: List[np.ndarray], prompt_text: str) -> str:
        """Feed sampled frames + prompt text into the VLM and get structured output."""
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": f} for f in video_frames],
                {"type": "text", "text": prompt_text}
            ]
        }]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        ).to(self.device)

        generation_config = {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "do_sample": False,
            "top_p": 1.0,
            "top_k": 1,
        }

        if self.deterministic:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        with torch.no_grad():
            outputs = self.vlm.generate(**inputs, **generation_config)
            trimmed = [o[len(i):].cpu() for i, o in zip(inputs.input_ids, outputs)]
            text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return text.strip()

    def parse_model_output(self, raw_output: str, vid_id: str, evaluation_questions: List) -> Dict:
        """Try to extract structured JSON from model output."""
        try:
            parsed = json.loads(raw_output)
            if "answers" in parsed:
                return parsed
        except Exception:
            pass

        # fallback
        start, end = raw_output.find("{"), raw_output.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(raw_output[start:end + 1])
                if "answers" in parsed:
                    return parsed
            except Exception:
                pass

        answers = {q["id"]: {"answer": "Unknown", "reason": "Failed to parse"} for q in evaluation_questions}
        return {"video_id": vid_id, "answers": answers, "parse_warning": "Parsing failed"}

    def calculate_weighted_score(self, answers: Dict, evaluation_questions: List, category: str) -> Dict:
        """Weighted average + reasoning-layer breakdown for CCR."""
        total_weight, weighted_sum = 0, 0
        question_scores = {}

        layer_keys = {
            "L1": "cultural_symbol_understanding",
            "L2": "contextual_alignment",
            "L3": "social_behavior_understanding"
        }

        layers = {
            layer_keys["L1"]: {"questions": [], "total_weight": 0.0, "weighted_sum": 0.0},
            layer_keys["L2"]: {"questions": [], "total_weight": 0.0, "weighted_sum": 0.0},
            layer_keys["L3"]: {"questions": [], "total_weight": 0.0, "weighted_sum": 0.0}
        }

        for q in evaluation_questions:
            qid, weight = q["id"], float(q["weight"])
            answer = answers.get(qid, {}).get("answer", "Unknown")
            score = 1.0 if answer.lower() == "yes" else 0.0
            weighted_score = score * weight

            weighted_sum += weighted_score
            total_weight += weight
            question_scores[qid] = {
                "question": q["question"],
                "answer": answer,
                "weight": weight,
                "score": score,
                "weighted_score": weighted_score
            }

            qnum = int(re.sub(r"\D", "", qid)) if re.search(r"\d+", qid) else 999
            if qnum <= 2:
                lname = layer_keys["L1"]
            elif 3 <= qnum <= 5:
                lname = layer_keys["L2"]
            else:
                lname = layer_keys["L3"]

            layers[lname]["questions"].append(qid)
            layers[lname]["weighted_sum"] += weighted_score
            layers[lname]["total_weight"] += weight

        weighted_avg = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

        layer_scores = {}
        for lname, ldata in layers.items():
            if ldata["total_weight"] > 0:
                layer_scores[lname] = {
                    "score": round(ldata["weighted_sum"] / ldata["total_weight"], 4),
                    "total_weight": ldata["total_weight"],
                    "weighted_sum": ldata["weighted_sum"],
                    "questions": ldata["questions"]
                }
            else:
                layer_scores[lname] = {"score": 0, "total_weight": 0, "weighted_sum": 0, "questions": []}

        return {
            "weighted_average_score": weighted_avg,
            "total_weight": round(total_weight, 4),
            "total_weighted_score": round(weighted_sum, 4),
            "question_scores": question_scores,
            "reasoning_layer_scores": layer_scores
        }

    def evaluate_video(self, json_data: Dict, video_path: str) -> Dict:
        """Main evaluation for one CCR video."""
        vid_id = json_data["id"]
        category = json_data.get("category", "Cultural Commonsense Reasoning")

        print(f"\n🚀 Evaluating video [{vid_id}] for category [{category}] ...")
        frames = self.extract_frames(video_path)
        print(f"🎞 Extracted {len(frames)} frames.")

        prompt_text = self.build_prompt_from_json_data(json_data)
        raw_output = self.ask_vlm(frames, prompt_text)
        print("🧠 Model raw output (trimmed):\n", raw_output[:300], "...")

        eval_qs = json_data["evaluation"]["evaluation_questions"]
        parsed = self.parse_model_output(raw_output, vid_id, eval_qs)
        scores = self.calculate_weighted_score(parsed["answers"], eval_qs, category)

        parsed.update({
            "scoring_results": scores,
            "video_path": video_path,
            "category": category,
            "timestamp": datetime.now().isoformat()
        })

        out_path = os.path.join(self.save_dir, f"ccr_{vid_id}_structured_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved result: {out_path}")
        print(f"📊 Overall Score: {scores['weighted_average_score']}")
        for lname, ldata in scores["reasoning_layer_scores"].items():
            print(f"  - {lname}: {ldata['score']}")
        return parsed


# ===== Example =====
"""
from evaluators.CCR_evaluator import CulturalReasoningEvaluator

evaluator = CulturalReasoningEvaluator(
    vlm_path="/mnt/workspace/cailingling/models/Qwen/Qwen3-VL-30B-A3B-Instruct",
    save_dir="./evaluation_results/ccr_v1"
)

result = evaluator.evaluate_video(
    json_data=ccr_001,
    video_path="/mnt/workspace/cailingling/cvpr2026/benchmark_exp2/generated_videos_multimodel/cogvideox/cultural_commonsense_reasoning/CCR_001_cogvideox.mp4"
)
"""
