import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.utils
from PIL import Image
from torchvision import models, transforms
import glob

class VISTDatasetImages(torch.utils.data.Dataset):
    def __init__(self, params):

        self.num_data_points_per_split = {}
        # load caption information for different splits
        self.subsets = ["train", "val", "test"]
        self.overfit = params["OVERFIT"]
        self.params = params
        self._split = "train"
        self._split_idx = 0

        distractor_paths = [
            params["DISTRACTOR_PATH_TRAIN"],
            params["DISTRACTOR_PATH_VAL"],
            params["DISTRACTOR_PATH_TEST"],
        ]
        self.distractors = self.process_distractors(distractor_paths)
        self.all_image_paths = []
        self.all_data = []
        for subset_id, subset in enumerate(self.subsets):
            self.num_data_points_per_split[subset] = len(self.distractors[subset_id])
            if params["OVERFIT"]:
                self.num_data_points_per_split[subset] = params[
                    "DATASET_SAMPLE_OVERFIT"
                ]
            img_root = os.path.join(self.params["IMG_ROOT"], subset)
            self.all_image_paths.append(set([os.path.basename(p) for p in glob.glob(os.path.join(img_root, "*"))]))


    def process_distractors(self, paths):
        all_distractors = []
        for p in paths:
            with open(p) as f:
                data = json.load(f)
                all_distractors.append(data)

        return all_distractors

    def __len__(self):
        return self.num_data_points_per_split[self._split]

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets
        self._split = split
        self._split_idx = self.subsets.index(split)

    @staticmethod
    def get_default_transform():
        return transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        """
        image indices: 5
        distractor image indices: 4
        """
        item = {}

        cur_distractors = self.distractors[self._split_idx]
        cur_sample = cur_distractors[index]
        cur_images = cur_sample["photo_sequence"]
        distractor_ids = cur_sample["distractors"]

        img_root = os.path.join(self.params["IMG_ROOT"], self._split)
        all_images = self.all_image_paths[self._split_idx]
        image_paths_jpg = [
            os.path.join(img_root, "%s.jpg" % img_id)
            for img_id in cur_images
        ]
        image_paths_png = [
            os.path.join(img_root, "%s.png" % img_id)
            for img_id in cur_images
        ]
        exists_path_jpg = [os.path.basename(p) in all_images for p in image_paths_jpg]
        exists_path_png = [os.path.basename(p) in all_images for p in image_paths_png]
        

        # check if paths exists
        if not all([exists_jpg or exists_png for exists_jpg, exists_png in zip(exists_path_jpg, exists_path_png)]):
            print("image not found")
            return
        transform = self.get_default_transform()
        # load images
        gt_images = []
        distractor_images = []
        for j in range(len(image_paths_jpg)):
            if exists_path_jpg[j]:
                p = image_paths_jpg[j]
            else:
                p = image_paths_png[j]
            try:
                loaded_image = transform(Image.open(p).convert("RGB"))
                gt_images.append(loaded_image)
            except:
                print("image corrupt")
                return
        gt_images = torch.stack(gt_images)

        distractor_image_paths_jpg = [
            os.path.join(img_root, "%s.jpg" % img_id)
            for img_id in distractor_ids
        ]
        distractor_image_paths_png = [
            os.path.join(img_root, "%s.png" % img_id)
            for img_id in distractor_ids
        ]
        distractor_exists_path_jpg = [os.path.basename(p) in all_images for p in distractor_image_paths_jpg]
        distractor_exists_path_png = [os.path.basename(p) in all_images for p in distractor_image_paths_png]

        if not all([exists_jpg or exists_png for exists_jpg, exists_png in zip(distractor_exists_path_jpg, distractor_exists_path_png)]):
            print("image not found")
            return

        for j in range(len(distractor_ids)):
            if distractor_exists_path_jpg[j]:
                p = distractor_image_paths_jpg[j]
            else:
                p = distractor_image_paths_png[j]
            try:
                loaded_image = transform(Image.open(p).convert("RGB"))
                distractor_images.append(loaded_image)
            except:
                print("image corrupt")
                return
        distractor_images = torch.stack(distractor_images)

        item["images"] = gt_images
        item["distractor_images"] = distractor_images

        item["id"] = torch.LongTensor([index])
        item["image_id"] = torch.LongTensor([int(c) for c in cur_images])
        item["distractor_image_ids"] = torch.LongTensor(
            [int(d) for d in distractor_ids]
            )

        return item