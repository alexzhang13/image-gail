import copy
import json
import os
import random
import time
from collections import namedtuple

import cv2
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
        root_dir = params["SIS_ROOT"]
        data_file = "%s.story-in-sequence.json"

        paths = [
            params["GENERATED_STORIES_PATH_TRAIN"],
            params["GENERATED_STORIES_PATH_VAL"],
            params["GENERATED_STORIES_PATH_TEST"],
        ]
        self.generated_stories_all = self.process_generated_stories(paths)

        distractor_paths = [
            params["DISTRACTOR_PATH_TRAIN"],
            params["DISTRACTOR_PATH_VAL"],
            params["DISTRACTOR_PATH_TEST"],
        ]
        self.distractors = self.process_distractors(distractor_paths)
        self.all_image_paths = []
        VIST_data = namedtuple(
            "VIST_data",
            "captions_storyline, images_storyline, albumid_2_idxs, images, imageid_2_idx, albumid_2_imageid, album_ids, image_ids",
        )
        self.all_data = []
        for subset_id, subset in enumerate(self.subsets):
            cur_data = json.load(open(os.path.join(root_dir, data_file % subset)))
            (
                captions_storyline,
                images_storyline,
                albumid_2_idxs,
            ) = self.process_annotations(cur_data)
            images, imageid_2_idx, albumid_2_imageid = self.process_images(cur_data)
            album_ids = sorted(albumid_2_imageid.keys())
            image_ids = sorted(imageid_2_idx.keys())
            self.all_data.append(
                VIST_data._make(
                    [
                        captions_storyline,
                        images_storyline,
                        albumid_2_idxs,
                        images,
                        imageid_2_idx,
                        albumid_2_imageid,
                        album_ids,
                        image_ids,
                    ]
                )
            )
            self.num_data_points_per_split[subset] = len(self.distractors[subset_id])
            if params["OVERFIT"]:
                self.num_data_points_per_split[subset] = params[
                    "DATASET_SAMPLE_OVERFIT"
                ]
            img_root = os.path.join(self.params["IMG_ROOT"], subset)
            self.all_image_paths.append(set([os.path.basename(p) for p in glob.glob(os.path.join(img_root, "*"))]))

    def process_annotations(self, data):
        captions_storyline = []
        images_storyline = []
        albumid_2_idxs = {}
        assert len(data["annotations"]) % 5 == 0

        for j in range(len(data["annotations"]) // 5):
            cur_annotations = data["annotations"][j * 5 : (j + 1) * 5]
            captions_storyline.append([])
            images_storyline.append([])

            albumid = cur_annotations[0][0]["album_id"]
            if albumid not in albumid_2_idxs:
                albumid_2_idxs[albumid] = []

            albumid_2_idxs[albumid].append(j)

            for ann in cur_annotations:
                captions_storyline[j].append(ann[0]["text"])
                images_storyline[j].append(ann[0]["photo_flickr_id"])

        return captions_storyline, images_storyline, albumid_2_idxs

    def process_distractors(self, paths):
        all_distractors = []
        for p in paths:
            with open(p) as f:
                data = json.load(f)
                all_distractors.append(data)

        return all_distractors

    def process_generated_stories(self, paths):
        all_gen_stories = []
        for p in paths:
            with open(p) as f:
                print("path", p)
                stories = json.load(f)
                cur_stories = {}
                for cur_story in stories["output_stories"]:
                    stories = cur_story["story_text_normalized"]
                    cur_stories["-".join(cur_story["photo_sequence"])] = stories
                all_gen_stories.append(cur_stories)
        return all_gen_stories

    def process_images(self, data):
        images = []
        imageid_2_idx = {}
        albumid_2_imageid = {}
        for j, image in enumerate(data["images"]):
            imageid_2_idx[image["id"]] = j
            if image["album_id"] not in albumid_2_imageid:
                albumid_2_imageid[image["album_id"]] = []
            albumid_2_imageid[image["album_id"]].append(image["id"])
            # prune the unnecessary attributes in the image metadata
            attributes = ["album_id", "url_o", "id"]
            pruned_image = {attr: image[attr] for attr in attributes if attr in image}
            images.append(pruned_image)

        return images, imageid_2_idx, albumid_2_imageid

    def __len__(self):
        return self.num_data_points_per_split[self._split]

    @property
    def split(self):
        return self._split

    @property
    def tokenizer(self):
        return self._tokenizer

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

    """
    Get list of urls given list of image ids in the VIST dataset.
    """

    def get_image_urls(self, image_ids):
        urls = []
        cur_data = self.all_data[self._split_idx]
        imageid_2_idx = cur_data.imageid_2_idx
        all_images = cur_data.images
        for i in image_ids:
            id = imageid_2_idx[str(i)]
            img = all_images[id]
            if "url_o" in img:
                urls.append(img["url_o"])
            else:
                urls.append("not found")
        return urls

    def __getitem__(self, index):
        """
        stories: 5 x len story
        captions: 5 x len caption
        contextual captions: 5 x len contextual captions
        image indices: 5
        distractors: 4 x len caption
        distractor image indices: 4
        load image meta data
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
            gt_images.append(transform(Image.open(p).convert("RGB")))
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
            distractor_images.append(transform(Image.open(p).convert("RGB")))
        distractor_images = torch.stack(distractor_images)

        item["images"] = gt_images
        item["distractor_images"] = distractor_images

        item["id"] = torch.LongTensor([index])
        item["image_id"] = torch.LongTensor([int(c) for c in cur_images])
        item["distractor_image_ids"] = torch.LongTensor(
            [int(d) for d in distractor_ids]
            )

        return item