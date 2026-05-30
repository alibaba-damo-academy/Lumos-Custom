#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluator for Dynamic Reference Frame (DRF)
"""

import os
import json
import re
import torch
import numpy as np
from typing import Dict, List
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
from decord import VideoReader, cpu
from datetime import datetime

class DynamicReferenceFrameEvaluator:
    """
    Generic evaluator (supports Dynamic Reference Frame and Occlusion Reasoning Integrity):
    1. Read structured evaluation JSON with dynamic number of questions
    2. Convert to prompt with multiple Yes/No questions
    3. Generate answers + reasoning using VLM
    4. Calculate weighted average score with 3-layer grouping (q1–2 / q3–5 / q6+)
    5. Save structured JSON result with score
    """

    def __init__(
        self,
        vlm_path: str = "/mnt/workspace/cailingling/models/Qwen/Qwen3-VL-30B-A3B-Instruct",
        save_dir: str = "./evaluation_results/drf_v1",
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
        """Sample frames from video (DRF and ORI)."""
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
        return [vr[i].asnumpy() for i in indices]

    def build_prompt_from_json_data(self, json_data: Dict) -> str:
        """Convert structured evaluation JSON data to LLM prompt text."""
        pid = json_data["id"]
        category = json_data.get("category", "Video Reasoning")
        prompt = json_data["prompt_en"]
        description = json_data.get("description", "")
        eval_info = json_data.get("evaluation", {})
        questions = eval_info.get("evaluation_questions", [])
        eval_description = eval_info.get("evaluation_description", "")

        q_text = "\n".join(
            [f"{i+1}. {q['question']} (weight: {q['weight']})" for i, q in enumerate(questions)]
        )

        instruction = eval_info.get("overall_instruction",
            "Answer 'Yes' or 'No' to each question and briefly explain your reasoning for each."
        )

        answers_structure = ",\n    ".join(
            [f'"{q["id"]}": {{"answer": "Yes or No", "reason": "..."}}' for q in questions]
        )

        prompt_text = f"""
You are an objective video reasoning evaluator specializing in the "{category}" metric.
You will analyze the video and answer the following binary (Yes/No) questions.

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
        """Feed frames + prompt to the VLM and get structured text output."""
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
        """Parse model output into JSON (variable number of questions)."""
        try:
            parsed = json.loads(raw_output)
            if "answers" in parsed and isinstance(parsed["answers"], dict):
                print("✅ Successfully parsed JSON directly")
                return parsed
            else:
                print("🔄 JSON parsed but missing required structure")
                raise json.JSONDecodeError("Missing required structure", "", 0)
        except json.JSONDecodeError as e:
            print(f"🔄 Method 1 failed: Direct JSON parsing - {e}")

        try:
            json_match = re.search(r'\{[^{}]*\{.*\}[^{}]*\}', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                if "answers" in parsed and isinstance(parsed["answers"], dict):
                    print("✅ Successfully extracted JSON with regex")
                    return parsed
                else:
                    print("🔄 JSON extracted but missing required structure")
                    raise json.JSONDecodeError("Missing required structure", "", 0)
        except json.JSONDecodeError:
            print("🔄 Method 2 failed: Regex extraction")

        start_idx = raw_output.find('{')
        end_idx = raw_output.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = raw_output[start_idx:end_idx+1]
            try:
                parsed = json.loads(json_str)
                if "answers" in parsed and isinstance(parsed["answers"], dict):
                    print("✅ Successfully extracted JSON with start/end markers")
                    return parsed
                else:
                    print("🔄 JSON extracted but missing required structure")
                    raise json.JSONDecodeError("Missing required structure", "", 0)
            except json.JSONDecodeError:
                print("🔄 Method 3 failed: Start/end marker extraction")

        print("🔄 All automated parsing methods failed, attempting improved manual construction")
        answers = {}
        for question in evaluation_questions:
            qid = question["id"]
            answers[qid] = {"answer": "Unknown", "reason": "Failed to parse model output"}

        result = {"video_id": vid_id, "answers": answers}

        for question in evaluation_questions:
            qid = question["id"]
            patterns = [
                rf'"{qid}"\s*:\s*{{\s*"answer"\s*:\s*"([^"]*)"',
                rf'"{qid}"\s*:\s*"([^"]*)"',
                rf'{qid}\s*:\s*([^\s,\.]+)',
                rf'question\s*{qid[1:]}\s*:\s*([^\s,\.]+)',
            ]
            found = False
            for ptn in patterns:
                try:
                    m = re.search(ptn, raw_output, re.IGNORECASE)
                    if m:
                        ans = m.group(1).strip().strip('"\'')
                        if ans.lower() in ["yes", "no"]:
                            result["answers"][qid]["answer"] = ans.capitalize()
                            found = True
                            break
                except Exception:
                    continue

            if not found:
                snippet = re.escape(question["question"][:60])
                m2 = re.search(snippet + r'.*?\b(Yes|No)\b', raw_output, re.IGNORECASE | re.DOTALL)
                if m2:
                    result["answers"][qid]["answer"] = m2.group(1).capitalize()

            reason_patterns = [
                rf'{qid}.*?"reason"\s*:\s*"([^"]*)"',
                rf'{qid}.*?reason[^:]*:\s*([^\n}}]*)',
                rf'{qid}.*?explanation[^:]*:\s*([^\n}}]*)',
            ]
            for rp in reason_patterns:
                try:
                    rm = re.search(rp, raw_output, re.IGNORECASE | re.DOTALL)
                    if rm:
                        reason_txt = rm.group(1).strip().strip('"\'')
                        reason_txt = re.sub(r'^[:,\s]*', '', reason_txt)
                        if reason_txt:
                            result["answers"][qid]["reason"] = reason_txt[:300]
                            break
                except Exception:
                    continue

        result["parse_warning"] = "Auto-generated from text due to JSON parsing failure"
        result["raw_output_preview"] = raw_output[:500] + "..." if len(raw_output) > 500 else raw_output
        return result

    def calculate_weighted_score(self, answers: Dict, evaluation_questions: List, category: str) -> Dict:
        """Weighted average and per-layer scores (q1–2 / q3–5 / q6+); layer names depend on category."""
        total_weight = 0.0
        weighted_sum = 0.0
        question_scores = {}

        if "Dynamic Reference Frame" in category:
            layer_keys = {
                "L1": "scene_viewpoint_consistency",      # q1–q2
                "L2": "relative_motion_coherence",       # q3–q5
                "L3": "physical_temporal_realism"        # q6+
            }
        elif "Occlusion Reasoning Integrity" in category:
            layer_keys = {
                "L1": "scene_occlusion_consistency",
                "L2": "structural_shape_continuity",
                "L3": "physical_depth_consistency"
            }
        else:
            layer_keys = {
                "L1": "layer_1",
                "L2": "layer_2",
                "L3": "layer_3"
            }

        reasoning_layers = {
            layer_keys["L1"]: {"questions": [], "total_weight": 0.0, "weighted_sum": 0.0},
            layer_keys["L2"]: {"questions": [], "total_weight": 0.0, "weighted_sum": 0.0},
            layer_keys["L3"]: {"questions": [], "total_weight": 0.0, "weighted_sum": 0.0}
        }

        for question in evaluation_questions:
            qid = question["id"]
            weight = float(question["weight"])
            answer_data = answers.get(qid, {})
            answer_text = str(answer_data.get("answer", "Unknown")).strip()

            score = 1.0 if answer_text.lower() == "yes" else 0.0
            q_weighted = score * weight
            weighted_sum += q_weighted
            total_weight += weight

            question_scores[qid] = {
                "question": question["question"],
                "weight": weight,
                "answer": answer_data.get("answer", "Unknown"),
                "score": score,
                "weighted_score": q_weighted
            }

            try:
                q_num = int(re.sub(r'\D', '', qid))
            except Exception:
                q_num = 999

            if q_num <= 2:
                layer_name = layer_keys["L1"]
            elif 3 <= q_num <= 5:
                layer_name = layer_keys["L2"]
            else:
                layer_name = layer_keys["L3"]

            reasoning_layers[layer_name]["questions"].append(qid)
            reasoning_layers[layer_name]["total_weight"] += weight
            reasoning_layers[layer_name]["weighted_sum"] += q_weighted

        weighted_average = (weighted_sum / total_weight) if total_weight > 0 else 0.0

        layer_scores = {}
        for lname, ldata in reasoning_layers.items():
            if ldata["total_weight"] > 0:
                layer_scores[lname] = {
                    "score": round(ldata["weighted_sum"] / ldata["total_weight"], 4),
                    "total_weight": round(ldata["total_weight"], 4),
                    "weighted_sum": round(ldata["weighted_sum"], 4),
                    "questions": ldata["questions"]
                }
            else:
                layer_scores[lname] = {
                    "score": 0.0, "total_weight": 0.0, "weighted_sum": 0.0, "questions": []
                }

        return {
            "weighted_average_score": round(weighted_average, 4),
            "total_weight": round(total_weight, 4),
            "total_weighted_score": round(weighted_sum, 4),
            "question_scores": question_scores,
            "reasoning_layer_scores": layer_scores
        }

    def evaluate_video(self, json_data: Dict, video_path: str) -> Dict:
        """Evaluate one video (DRF or ORI)."""
        vid_id = json_data["id"]
        category = json_data.get("category", "Video Reasoning")
        print(f"🚀 Evaluating video [{vid_id}] for category [{category}] ...")

        prompt_text = self.build_prompt_from_json_data(json_data)

        frames = self.extract_frames(video_path)
        print(f"🎞 Extracted {len(frames)} frames from video.")

        raw_output = self.ask_vlm(frames, prompt_text)
        print("🧠 Model raw output:\n", raw_output)

        evaluation_questions = json_data["evaluation"]["evaluation_questions"]

        parsed = self.parse_model_output(raw_output, vid_id, evaluation_questions)

        score_result = self.calculate_weighted_score(parsed.get("answers", {}), evaluation_questions, category)

        parsed["scoring_results"] = score_result
        parsed["video_id"] = vid_id
        parsed["video_path"] = video_path
        parsed["category"] = category
        parsed["timestamp"] = datetime.now().isoformat()
        parsed["raw_output_length"] = len(raw_output)

        parsed["evaluation_questions_info"] = [
            {"id": q["id"], "question": q["question"], "weight": q["weight"]}
            for q in evaluation_questions
        ]

        safe_category = re.sub(r'\W+', '_', category.lower())
        out_path = os.path.join(self.save_dir, f"{safe_category}_{vid_id}_structured_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to: {out_path}")
        print(f"📊 Overall weighted average score: {score_result['weighted_average_score']}")

        layer_scores = score_result["reasoning_layer_scores"]
        print("📈 Reasoning Layer Scores:")
        for lname, ldata in layer_scores.items():
            print(f"  - {lname}: {ldata['score']}")

        return parsed


"""
Example (DRF_001):
drf_001 = {
  "id": "DRF_001",
  "category": "Dynamic Reference Frame",
  "prompt_en": "A person looks out from a moving train at trees passing by in the opposite direction.",
  "description": "Tests whether the model can represent relative motion between the observer and the external world.",
  "difficulty": "easy",
  "evaluation": {
    "overall_instruction": "Answer 'Yes' or 'No' to each question and briefly explain your reasoning for each.",
    "evaluation_description": "Three reasoning layers are assessed: (1) Scene & viewpoint consistency (q1–q2). (2) Relative motion coherence (q3–q5). (3) Physical & temporal realism (q6–q8).",
    "evaluation_questions": [
      {"id": "q1", "question": "Is the viewpoint clearly located inside a moving train (e.g., visible interior or stable window frame)?", "weight": 1},
      {"id": "q2", "question": "Is a person visible looking out the window toward the outside scenery?", "weight": 1},
      {"id": "q3", "question": "Do the trees or other outside objects move in the opposite direction to the train, showing correct relative motion?", "weight": 3},
      {"id": "q4", "question": "Is the speed of outside scenery consistent with realistic train velocity (smooth and continuous)?", "weight": 2},
      {"id": "q5", "question": "Does the parallax effect appear correct — nearby trees move faster than distant ones, indicating correct depth reasoning?", "weight": 3},
      {"id": "q6", "question": "Does the interior of the train and the person remain stable without unintended shaking or distortion?", "weight": 2},
      {"id": "q7", "question": "Are motion directions of the train and outside scenery consistent through time, without reversal or sudden changes?", "weight": 2},
      {"id": "q8", "question": "Do lighting and reflections (e.g., on the window) stay coherent with the direction of motion?", "weight": 1}
    ],
    "scoring_method": "weighted_average"
  }
}

# evaluator = DynamicReferenceFrameEvaluator(
#     vlm_path="/mnt/workspace/cailingling/models/Qwen/Qwen3-VL-30B-A3B-Instruct",
#     save_dir="./evaluation_results/drf_v1"
# )
# result = evaluator.evaluate_video(drf_001, "/path/to/video.mp4")
"""
