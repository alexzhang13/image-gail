import numpy as np
import pandas as pd 
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from dataloader.vist_images_dataloader import VISTDatasetImages
from utils.helper_utils import prune_illegal_collate, batch_iter

class Expert():
    params = {
        "BATCH_PER_GPU": 16,
        "DATASET_SAMPLE_OVERFIT": 100,
        "DISTRACTOR_PATH_TEST": "configs/vist_distractor_ids/seed-123/test_hard_sampled_distractors.json",
        "DISTRACTOR_PATH_TRAIN": "configs/vist_distractor_ids/seed-123/train_hard_sampled_distractors.json",
        "DISTRACTOR_PATH_VAL": "configs/vist_distractor_ids/seed-123/val_hard_sampled_distractors.json",
        "IMG_ROOT": "/n/fs/nlp-murahari/datasets/VIST/images",
        "NUM_DISTRACTORS": 4,
        "OVERFIT": False,
    }

    def __init__(self, bs, nepochs):
        self.epochs = nepochs
        self.vist_dataset_images = VISTDatasetImages(params)
        self.dataloader = DataLoader(
            vist_dataset_images,
            batch_size=bs,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=False,
            collate_fn=prune_illegal_collate,
        )

    def sample(self):
        for _, iter_id, batch in batch_iter(dataloader, self.epochs):
            print("iter id", iter_id)