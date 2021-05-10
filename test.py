import argparse
import numpy as np
import torch 
import torch.nn as nn 
import torchvision
from torch.utils.data import DataLoader

from models.gail import Discriminator, Policy, Gail
from models.expert import Expert

from dataloader.vist_images_dataloader import VISTDatasetImages
from utils.helper_utils import prune_illegal_collate, batch_iter

torch.cuda.empty_cache()

# add variables here for parser later
parser = argparse.ArgumentParser(description="Arguments for training")
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--path', default="./saved_models/checkpoint_epoch_1.t7")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# init random seeding
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# dataloader params
seq_length = 5
num_distractors = 4
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

def test_loop():    
    # initialize env and expert trajectories
    freeze_resnet = True
    curr_epoch_id = 0
    vist_dataset_images = VISTDatasetImages(params)
    vist_dataset_images.split = "test"
    dataloader = DataLoader(
        vist_dataset_images,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=False,
        collate_fn=prune_illegal_collate,
    )

    # initialize models
    agent = Gail(input_dim=(2*2048), lr=args.lr, seq_length=seq_length, device=device)
    agent.load(args.path)

    # Evaluation
    for _, iter_id, batch in batch_iter(dataloader, 1):
        batch_raw = batch['images']
        batch_size = batch_raw.shape[0]
        batch_raw = torch.FloatTensor(batch_raw).to(device)

        distractors = batch['distractor_images']
        distractors = torch.FloatTensor(distractors).to(device)
        distractors = torch.reshape(distractors, (batch_size*num_distractors, distractors.shape[2], distractors.shape[3], distractors.shape[4])) 

        feat_distractors = agent.resnet(distractors)
        feat_distractors = torch.reshape(feat_distractors, (batch_size,num_distractors,-1))

        # sample trajectories
        references = [] 
        for i in range(seq_length):
            imgs = agent.resnet(batch_raw[:,i]) 
            imgs = torch.reshape(imgs, (batch_size,-1))
            references.append(imgs)

        # compare candidates and reference images
        correct = 0
        for i in range(seq_length-1):
            imgs = references[i]
            refs = references[i+1]
            action = agent.policy(imgs)
            preds = torch.normal(action, 0.01)

            # reshape for concatenation
            preds = torch.unsqueeze(preds, dim=1)
            refs = torch.unsqueeze(refs, dim=1)
            candidates = torch.cat([refs, feat_distractors], dim=1)

            preds = torch.repeat_interleave(preds, num_distractors+1, dim=1)
            feat_diff = torch.norm(preds - candidates, p=2, dim=2)
            min_indices = torch.argmin(feat_diff, dim=1).flatten()
            zeros = min_indices == 0
            correct += zeros.nonzero().shape[0]
        
        accuracy = correct / (batch_size)

    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    test_loop()