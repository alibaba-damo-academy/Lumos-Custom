# -*- coding:utf-8 -*-

# ==============================================================================
#
#           Video Light and Shadow Analysis Pipeline
#
# ==============================================================================
# This script defines the core logic for analyzing video files to extract
# both quantitative and qualitative lighting attributes. It consists of three
# main components:
#
# 1. `read_video_frames`: A utility function to efficiently sample frames
#    from a video file and prepare them for analysis.
#
# 2. `parse_response2json`: A helper function to clean and parse the raw
#    string output from the language model into a structured JSON format.
#
# 3. `VideoLightAndShadowPredictor`: The main class that orchestrates the
#    entire pipeline. It loads a Vision-Language Model (VLLM), processes
#    a given video, performs inference to get qualitative descriptions,
#    calculates quantitative metrics, and combines them into a single,
#    structured output.
# ==============================================================================


import os
import json

import cv2
import numpy as np

from ._vllm_model import VllmQwen2VLModel
from ._prompts import SYSTEM_PROMPT, PROMPT
from ._quantitative_utils import calculate_image_cct_and_illuminance


def read_video_frames(path, num_frames=16, max_resolution=(1080, 1920)):
    """
    Reads and samples frames from a video file.

    This function opens a video, samples a specified number of frames uniformly
    across its duration, and resizes them if they exceed a maximum resolution.
    It returns both the stack of sampled frames and the single middle frame, which
    is often used as a representative image for static analysis.

    Args:
        path (str): The file path to the video.
        num_frames (int): The target number of frames to sample. If non-positive or
                          greater than total frames, all frames are read.
        max_resolution (tuple): A tuple (height, width) specifying the maximum
                                allowed resolution for frames.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: A NumPy array of shape (num_frames, height, width, 3)
                          containing the sampled video frames in BGR format.
            - np.ndarray: The single middle frame from the sampled sequence.
    
    Raises:
        ValueError: If the video file cannot be opened, is empty, or if no
                    frames can be extracted.
    """
    # Initialize a video capture object from the specified path.
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    # Get the total number of frames in the video.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"The video file is empty or corrupted: {path}")

    # Determine the indices of the frames to be extracted.
    if num_frames <= 0 or num_frames >= total_frames:
        # If an invalid number of frames is requested, default to sampling all frames.
        frame_indices = set(range(total_frames))
    else:
        # Calculate uniform intervals to sample frames across the video's duration.
        interval = total_frames / num_frames
        frame_indices = set(int(i * interval) for i in range(num_frames))

    frames = []
    current_frame_idx = 0
    max_height, max_width = max_resolution

    # Loop through the video to read frames.
    while current_frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            # If a frame cannot be read, break the loop.
            print(f"Warning: Cannot read frame {current_frame_idx} from {path}")
            break

        if current_frame_idx in frame_indices:
            # --- Resize frame if it exceeds the maximum resolution ---
            # This helps to standardize input size and prevent memory issues.
            height, width = frame.shape[:2]
            scale = 1.0

            if max_height is not None and height > max_height:
                scale = min(scale, max_height / height)
            if max_width is not None and width > max_width:
                scale = min(scale, max_width / width)

            if scale < 1.0:
                new_size = (int(width * scale), int(height * scale))
                # cv2.INTER_AREA is generally recommended for downsampling.
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

            frames.append(frame)
            
            # Optimization: if we have collected the desired number of frames, stop early.
            if num_frames > 0 and len(frames) >= num_frames:
                break

        current_frame_idx += 1

    # Always release the capture object to free up resources.
    cap.release()

    if not frames:
        raise ValueError(f"Could not extract any frames from video: {path}")

    # Convert the list of frames into a single NumPy array for efficient processing.
    frames_array = np.stack(frames)

    # Select the middle frame of the sequence as a representative image.
    middle_frame_index = len(frames_array) // 2

    return frames_array, frames_array[middle_frame_index]


def parse_response2json(raw_response_str):
    """
    Cleans and parses a raw string response from an LLM into a JSON object.

    The LLM might return a string wrapped in markdown code fences (e.g., ```json...```)
    and with inconsistent formatting. This function standardizes and parses it.

    Args:
        raw_response_str (str): The raw string output from the language model.

    Returns:
        dict: A Python dictionary parsed from the cleaned JSON string.
    """
    # Standardize formatting: remove newlines and use consistent commas.
    cleaned_str = raw_response_str.replace("\n", "").replace("，", ",")
    # Strip potential markdown code block fences (e.g., "```json" at the start and "```" at the end).
    json_part = cleaned_str[7:-3]
    # Parse the cleaned string into a Python dictionary.
    parsed_data = json.loads(json_part)
    return parsed_data


class VideoLightAndShadowPredictor:
    """
    A class to orchestrate the video light and shadow prediction pipeline.

    It initializes a Vision-Language Model and provides a simple `inference`
    method to process a video file and return its lighting attributes.
    """

    def __init__(self, weight_path, device='cuda'):
        """
        Initializes the predictor.

        Args:
            weight_path (str): The directory path containing the model weights.
            device (str): The device to run the model on (e.g., 'cuda' or 'cpu').
        """
        self.weight_path = weight_path
        self.device = device
        
        # Initialize the Vision-Language Model (VLLM).
        # This is a heavyweight operation and is performed only once during object creation.
        self._model = VllmQwen2VLModel(model_fp=os.path.join(self.weight_path, 'Qwen2.5-VL-7B-Instruct'),
                                       device=self.device)

    def inference(self, local_file_path):
        """
        Performs inference on a single video file.

        Args:
            local_file_path (str): The path to the local video file.

        Returns:
            dict or None: A dictionary containing the analysis results
                          (cct, illuminance, xy, qualitative) or None if
                          any step in the pipeline fails.
        """
        # --- Step 1: Read and preprocess video frames ---
        try:
            # Sample frames and get the middle frame for quantitative analysis.
            extracted_frames, mid_frame = read_video_frames(local_file_path, num_frames=16)
            # Convert frames from BGR (OpenCV's default) to RGB, which is expected by most DL models.
            full_frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in extracted_frames]
        except Exception as e:
            print(f"Error opening or processing video {local_file_path}: {e}. Skipping.")
            return None

        # --- Step 2: Perform qualitative analysis with the VLLM ---
        # The model is fed the video frames and structured prompts to generate a text description.
        model_res = self._model(video_frames_pil_rgb=full_frames_rgb,
                                modality="video",
                                system_prompt=SYSTEM_PROMPT,
                                question=PROMPT)
        
        # --- Step 3: Perform quantitative analysis on the middle frame ---
        try:
            # Calculate numerical metrics like Correlated Color Temperature (CCT) and Illuminance.
            cct, illuminance, xy = calculate_image_cct_and_illuminance(mid_frame)
        except Exception as e:
            print(f"Error calculating quantitative metrics for {local_file_path}: {e}. Skipping.")
            return None

        # --- Step 4: Parse the LLM's raw text response ---
        try:
            # Convert the raw string from the model into a structured dictionary.
            parsed_data = parse_response2json(model_res[0])
        except Exception as e:
            print(f"Error parsing LLM response for {local_file_path}: {e}. Skipping.")
            return None

        # --- Step 5: Combine all results into a final dictionary ---
        # Convert NumPy array to a standard Python list to ensure it's JSON serializable.
        xy_list = xy.tolist()
        res_dict = dict(cct=cct, illuminance=illuminance, xy=xy_list, qualitative=parsed_data)

        return res_dict
