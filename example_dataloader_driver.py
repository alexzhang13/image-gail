# get config file
from dataloader.vist_images_dataloader import VISTDatasetImages
from utils.helper_utils import prune_illegal_collate, batch_iter
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
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
    vist_dataset_images = VISTDatasetImages(params)
    bs = 16
    dataloader = DataLoader(
        vist_dataset_images,
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=False,
        collate_fn=prune_illegal_collate,
    )
    for _, iter_id, batch in batch_iter(dataloader, 1):
        print("iter id", iter_id)
