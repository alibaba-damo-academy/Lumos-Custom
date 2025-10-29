# -*- coding:utf-8 -*-
import numpy as np

from transformers import AutoProcessor
from vllm import LLM, SamplingParams


class VllmQwen2VLModel:
    def __init__(self, model_fp, device):
        self.llm = LLM(
            model=model_fp,
            max_model_len=32786,  # 16384
            gpu_memory_utilization=0.8)
            # limit_mm_per_prompt={"image": 1, "video": 1})
        self.processor = AutoProcessor.from_pretrained(model_fp)

        self.sampling_params = SamplingParams(
            temperature=0.5,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=1024,
            stop_token_ids=None)

    @staticmethod
    def get_prompt(modality, system_prompt, question):
        if modality == "image":
            placeholder = "<|image_pad|>"
        elif modality == "video":
            placeholder = "<|video_pad|>"
        else:
            raise NotImplementedError

        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n")

        return prompt
    
    def __call__(self,
                 video_frames_pil_rgb,
                 modality,
                 system_prompt="",
                 question="What is the content of this image?"):
        prompt = VllmQwen2VLModel.get_prompt(
            modality=modality, 
            system_prompt=system_prompt, 
            question=question)

        # llm_inputs = [{
        #     "prompt": prompt,
        #     "multi_modal_data": {
        #         modality: np.array(_)
        #     }
        # } for _ in video_frames_pil_rgb]
        # print(len(llm_inputs))
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                modality: np.array(video_frames_pil_rgb)
            }
        }

        outputs = self.llm.generate(llm_inputs,
                                    sampling_params=self.sampling_params,
                                    use_tqdm=True)
        
        results = []
        for _ in outputs:
            results.append(_.outputs[0].text)

        return results