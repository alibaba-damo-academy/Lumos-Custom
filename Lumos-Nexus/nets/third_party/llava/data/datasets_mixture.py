# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})


DATASETS = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():
    promptfix_1m = Dataset(
        dataset_name="promptfix_1m",
        dataset_type="unified_gen",
        data_path="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/zhangpingzhe_data/Generate_data_in_unified_state/Image_Editing/promptFix/promptFix-1M-Format.json",
        image_path="",
        description="1M image editing data from PromptFix.",
    )
    add_dataset(promptfix_1m)

    hive_1m = Dataset(
        dataset_name="hive_1m",
        dataset_type="unified_gen",
        data_path="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/zhangpingzhe_data/Generate_data_in_unified_state/Image_Editing/HIVE/data2/HIVE-1.1M-Format.json",
        image_path="",
        description="1M image editing data from HIVE.",
    )
    add_dataset(hive_1m)

    ultraedit_4m = Dataset(
        dataset_name="ultraedit_4m",
        dataset_type="unified_gen",
        data_path="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/zhangpingzhe_data/Generate_data_in_unified_state/Image_Editing/UltraEdit/UltraEdit-4M-format.json",
        image_path="",
        description="4M image editing data from UltraEdit.",
    )
    add_dataset(ultraedit_4m)

    anyedit_2m = Dataset(
        dataset_name="anyedit_2m",
        dataset_type="unified_gen",
        data_path="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/zhangpingzhe_data/Generate_data_in_unified_state/Image_Editing/Bin1117/AnyEdit/Anyedit/Anyedit2.5M_format.json",
        image_path="",
        description="4M image editing data from UltraEdit.",
    )
    add_dataset(anyedit_2m)

    journey_4m = Dataset(
        dataset_name="journey_4m",
        dataset_type="unified_gen",
        data_path="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/zhangpingzhe_data/Generate_data_in_unified_state/Single_picture_text_data/JourneyDB/JourneyDB/check1/JourneyDB_train_4.2M_Format.json",
        image_path="",
        description="4M image editing data from UltraEdit.",
    )
    add_dataset(journey_4m)

    llava_1m = Dataset(
        dataset_name="llava_1m",
        dataset_type="unified_gen",
        data_path="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/zhangpingzhe_data/Generate_data_in_unified_state/Single_picture_text_data/liuhaotian/LLaVA1.1M_reformatted.json",
        image_path="",
        description="4M image editing data from UltraEdit.",
    )
    add_dataset(llava_1m)

    sharegp4v_1m = Dataset(
        dataset_name="sharegp4v_1m",
        dataset_type="unified_gen",
        data_path="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/zhangpingzhe_data/Generate_data_in_unified_state/Single_picture_text_data/Lin-Chen/SHARE1.39M_reformatted.json",
        image_path="",
        description="4M image editing data from UltraEdit.",
    )
    add_dataset(sharegp4v_1m)

    cambrian_5m = Dataset(
        dataset_name="cambrian_5m",
        dataset_type="unified_gen",
        data_path="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/zhangpingzhe_data/Generate_data_in_unified_state/Single_picture_text_data/nyu-visionx/Cambrian-10M/jsons/Cambrian-5.3M-format.json",
        image_path="",
        description="4M image editing data from UltraEdit.",
    )
    add_dataset(cambrian_5m)

    mmc4_core_ff_5m = Dataset(
        dataset_name="mmc4_core_ff_5m",
        dataset_type="unified_gen",
        data_path="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/zhangpingzhe_data/Generate_data_in_unified_state/Multiple_pictures_text_data/MMC4/MMC4-core-ff_5M_format.json",
        image_path="",
        description="4M image editing data from UltraEdit.",
    )
    add_dataset(mmc4_core_ff_5m)

    data_v1_filtered_llava_13b_captioned_10m = Dataset(
        dataset_name="data_v1_filtered_llava_13b_captioned_10m",
        dataset_type="unified_gen",
        data_path="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/qlz/projects/unified_gen/data_v1_filtered_llava_13b_captioned_10m.json",
        image_path="",
        description="4M image editing data from UltraEdit.",
    )
    add_dataset(data_v1_filtered_llava_13b_captioned_10m)