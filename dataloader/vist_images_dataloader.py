import copy
import json
import os
import random
import time
import imagesize
import numpy as np
import torch
import torch.utils
from PIL import Image
from torchvision import models, transforms
import glob
import lmdb

class VISTDatasetImages(torch.utils.data.Dataset):
    def __init__(self, params):

        self.num_data_points_per_split = {}
        # load caption information for different splits
        self.subsets = ["train", "val", "test"]
        self.params = params
        self._split = "train"
        self._split_idx = 0
        self.envs = []

        distractor_paths = [
            params["DISTRACTOR_PATH_TRAIN"],
            params["DISTRACTOR_PATH_VAL"],
            params["DISTRACTOR_PATH_TEST"],
        ]
        self.distractors = self.process_distractors(distractor_paths)
        for subset_id, subset in enumerate(self.subsets):
            self.num_data_points_per_split[subset] = len(self.distractors[subset_id])
            # initialize lmdb
            path_to_lmdb = os.path.join(params["IMG_LMDB_ROOT"], subset)
            self.envs.append(lmdb.open(path_to_lmdb, readonly=True, lock=False, map_size=int(1e12)))
            

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
        env = self.envs[self._split_idx]

        distractor_images = []
        gt_images = []

        with env.begin() as e:

            for distractor in distractor_ids:
                    img_binary = e.get(distractor.encode())
                    if img_binary is None:
                        print("image not found in database", distractor)
                        return
                    try:
                        img = np.frombuffer(img_binary, dtype="float32").reshape(3, 256, 256)
                        img = torch.from_numpy(img)
                        distractor_images.append(img)
                    except:
                        print("corrupted image", distractor)
                        return

            distractor_images = torch.stack(distractor_images)

            for cur_image in cur_images:
                with env.begin() as e:
                    img_binary = e.get(cur_image.encode())
                    if img_binary is None:
                        print("image not found in database", cur_image)
                        return
                    try:
                        img = np.frombuffer(img_binary, dtype="float32").reshape(3, 256, 256)
                        img = torch.from_numpy(img)
                        gt_images.append(img)
                    except:
                        print("corrupted image", cur_image)
                        return

            gt_images = torch.stack(gt_images)            

        item["images"] = gt_images
        item["distractor_images"] = distractor_images

        item["id"] = torch.LongTensor([index])
        item["image_id"] = torch.LongTensor([int(c) for c in cur_images])
        item["distractor_image_ids"] = torch.LongTensor(
            [int(d) for d in distractor_ids]
            )

        return item