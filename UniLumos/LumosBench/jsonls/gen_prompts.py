# -*- coding: utf-8 -*-

# ==============================================================================
#
#               Modular Dataset Generation for Video Relighting
#
# ==============================================================================
#
# ## Principle of Operation
#
# This script implements a modular and scalable approach to generate a large,
# structured dataset for video relighting tasks. The core challenge is that a
# full combinatorial explosion of all possible scene descriptions and all
# possible lighting conditions would result in an impractically large dataset.
# For instance, with 49 source videos and 2,268 unique lighting combinations,
# a full cartesian product would yield over 111,000 data points.
#
# To manage this complexity, we adopt a two-step, modular strategy:
#
# 1.  **Decoupled Generation of Sources:**
#     - **Video Source:** A base set of video metadata (e.g., scene description,
#       resolution) is pre-processed into a clean JSONL file (`video_source_file`).
#       Each line represents a unique video scene.
#     - **Lighting Source:** A separate script generates a comprehensive "library"
#       of all possible lighting condition combinations (`lighting_source_file`).
#       Each line in this file represents a unique, structured lighting setup.
#
# 2.  **Stochastic Sampling and Pairing:**
#     - This script acts as the second stage. Instead of creating all possible
#       pairings, it performs a stochastic (random) sampling process.
#     - It first randomly selects a small subset of videos from the video source.
#     - Then, for each selected video, it randomly samples a specified number of
#       lighting conditions from the lighting library.
#     - Finally, it pairs the video metadata with the sampled lighting metadata
#       to create the final dataset entries.
#
# ## Advantages of this Modular Approach
#
# -   **Scalability:** We can easily control the size of the final dataset by
#     adjusting the sampling parameters (`num_videos`, `num_lights_per_video`)
#     without regenerating the source files.
# -   **Reusability:** The `lighting_combinations.jsonl` file becomes a reusable
#     asset for various experiments. We can generate different datasets (e.g.,
#     a 2k, 5k, or 10k dataset) from the same source files.
# -   **Reproducibility:** By using a fixed random seed, we ensure that the
#     exact same sampled dataset can be regenerated, which is crucial for
#     reproducible research.
# -   **Clarity:** The logic is separated into distinct, manageable steps, making
#     the code easier to understand, debug, and maintain.
#
# This script, therefore, serves as the final assembler, creating a targeted,
# reasonably-sized, and reproducible dataset from two independent, comprehensive
# source pools.
# ==============================================================================


import json
import random

def create_sampled_dataset_v2(video_file, lighting_file, output_file, num_videos, num_lights_per_video, seed=42):
    """
    Randomly samples from a video source file and a lighting source file,
    pairs them, and generates a final dataset. This version also creates a
    concatenated 'new_text' field for direct use.

    Args:
        video_file (str): Path to the JSONL file containing video metadata.
        lighting_file (str): Path to the JSONL file containing all lighting combinations.
        output_file (str): Path for the final output dataset file.
        num_videos (int): The number of videos to randomly sample from the source.
        num_lights_per_video (int): The number of lighting conditions to pair with each sampled video.
        seed (int): A random seed to ensure reproducibility of the sampling process.
    """
    # Set the random seed to ensure that the sampling is deterministic.
    # This means running the script multiple times with the same seed will produce the exact same output file.
    random.seed(seed)
    
    try:
        # --- Step 1: Load all source data into memory ---
        # This is feasible for moderately sized source files. For extremely large files,
        # a line-by-line reading approach or a database would be more memory-efficient.
        print("Loading source data...")
        with open(video_file, 'r', encoding='utf-8') as f:
            videos_data = [json.loads(line) for line in f]
        
        with open(lighting_file, 'r', encoding='utf-8') as f:
            lighting_data = [json.loads(line) for line in f]
        
        print(f"Loading complete: {len(videos_data)} video metadata records, {len(lighting_data)} lighting combinations.")

        # --- Sanity Checks: Ensure sampling is possible ---
        # Verify that the requested number of samples does not exceed the available data.
        if num_videos > len(videos_data):
            print(f"Error: Number of videos to sample ({num_videos}) is greater than available videos ({len(videos_data)}).")
            return
        if num_lights_per_video > len(lighting_data):
            print(f"Error: Number of lights to sample ({num_lights_per_video}) is greater than available combinations ({len(lighting_data)}).")
            return
            
        # --- Step 2: Randomly sample a subset of videos ---
        # `random.sample` performs sampling without replacement, ensuring each selected video is unique.
        sampled_videos = random.sample(videos_data, num_videos)
        print(f"\nRandomly sampled {len(sampled_videos)} videos.")

        # --- Step 3: Pair each sampled video with a random subset of lighting conditions ---
        final_dataset = []
        print(f"Pairing each video with {num_lights_per_video} random lighting combinations...")
        
        for video_meta in sampled_videos:
            # For each video, sample a new set of random lighting conditions.
            sampled_lights = random.sample(lighting_data, num_lights_per_video)
            
            for light_meta in sampled_lights:
                # Merge the video metadata and the lighting metadata into a single record.
                # The dictionary unpacking syntax `{**a, **b}` is a concise way to merge dictionaries.
                # If any keys overlap, the values from the second dictionary (`light_meta`) would take precedence.
                combined_record = {**video_meta, **light_meta}
                
                # --- New Step: Create the final concatenated text prompt ---
                # Safely get the original scene description text. Defaults to empty string if 'text' key is missing.
                scene_text = combined_record.get('text', '')
                # Safely get the lighting description, and remove any trailing period to prevent "text..".
                light_prompt_text = combined_record.get('lighting_prompt', '').rstrip('.')
                
                # Concatenate the scene description and lighting description into a single prompt string.
                # This 'new_text' field is formatted to be directly consumable by a downstream model.
                combined_record['new_text'] = f"{scene_text} {light_prompt_text}."
                
                final_dataset.append(combined_record)
        
        # --- Step 4: Write the final dataset to the output file ---
        print("\nWriting the final dataset to file...")
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for record in final_dataset:
                # `json.dumps` serializes the Python dictionary into a JSON formatted string.
                # `ensure_ascii=False` is important for correctly handling non-ASCII characters.
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

        total_records = len(final_dataset)
        print("\nProcessing complete!")
        print(f"Successfully generated {total_records} records. ({num_videos} videos * {num_lights_per_video} lights/video)")
        print(f"Final dataset saved to: {output_file}")

    except FileNotFoundError as e:
        print(f"Error: A source file was not found - {e}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")


if __name__ == '__main__':
    # --- Configuration Parameters ---
    # This section makes it easy to change file paths and sampling settings without touching the function logic.
    
    # Input files
    VIDEO_SOURCE_FILE = 'test_gen_videos.jsonl'       # The source file with base video metadata.
    LIGHTING_SOURCE_FILE = 'lighting_combinations.jsonl' # The "library" of all possible lighting conditions.
    
    # Output file
    FINAL_OUTPUT_FILE = 'final_dataset_2k.jsonl'     # The destination for the generated dataset.
    
    # Sampling settings
    NUM_VIDEOS_TO_SAMPLE = 10  # How many unique videos to choose.
    NUM_LIGHTS_PER_VIDEO = 200 # How many lighting conditions to pair with each video.
    RANDOM_SEED = 42           # Use a constant seed for reproducible results.
    
    # --- Execute the Function ---
    create_sampled_dataset_v2(
        VIDEO_SOURCE_FILE,
        LIGHTING_SOURCE_FILE,
        FINAL_OUTPUT_FILE,
        NUM_VIDEOS_TO_SAMPLE,
        NUM_LIGHTS_PER_VIDEO,
        RANDOM_SEED
    )
