import argparse
import math
import numpy as np
import torch 
import torch.nn as nn 
import torchvision
import pdb
import os
from torch.utils.data import DataLoader
import logging
import json

from models.gail import Discriminator, Policy, Gail

from dataloader.vist_images_dataloader import VISTDatasetImages
from utils.helper_utils import prune_illegal_collate, batch_iter
from datetime import datetime

torch.cuda.empty_cache()

# add variables here for parser later
parser = argparse.ArgumentParser(description="Arguments for training")
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--variance', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--name', default="")
parser.add_argument('--path', default="./saved_models/checkpoint_epoch_20.t7")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Test name: ", args.name)
print("Device:", device)

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
        "IMG_LMDB_ROOT": "/n/fs/nlp-murahari/datasets/VIST/images/image-feats",
        "NUM_DISTRACTORS": 4,
        "OVERFIT": False,
    }

def test_loop():    
    # initialize env and expert trajectories
    vist_dataset_images = VISTDatasetImages(params)
    vist_dataset_images.split = "val"

    if not os.path.exists("./logger/"):
        os.makedirs("./logger/")

    logging.basicConfig(level=logging.DEBUG, filename="./logger/test_" + args.name + ".log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    # initialize models
    agent = Gail(input_dim=(2*2048), lr=args.lr, seq_length=seq_length, device=device)
    agent.load_checkpoint(args.path)
    print("Using ", torch.cuda.device_count(), "GPUs")

    # Evaluation
    print("time:%s \t Testing Loop Begins"%(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
    evaluation(agent, 0, vist_dataset_images, True)
    evaluation(agent, 0, vist_dataset_images, False)

def normal(action, action_prob, sigma):
    exponent = -0.5 * torch.pow((action - action_prob)/sigma, 2)
    f = 1/(math.sqrt(2 * sigma * math.pi)) * torch.exp(exponent)
    log_probs = torch.log(f)
    prob = torch.sum(log_probs, axis=1)
    return prob

def evaluation(agent, epoch_id, vist_dataset_images, val):
    if val:
        vist_dataset_images.split = "val"
    else:
        vist_dataset_images.split = "test"

    dl = DataLoader(
        vist_dataset_images,
        batch_size=2 * args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=False,
        pin_memory=False,
        collate_fn=prune_illegal_collate,
    )
    full_data = []
    agent.eval_mode()
    with torch.no_grad():
        correct = 0
        r3_correct = 0
        total = 0
        full_data = []
        for _, _iter_id, _batch in batch_iter(dl, 1):
            batch_raw = _batch['images']
            batch_size = batch_raw.shape[0]
            batch_raw = torch.FloatTensor(batch_raw).to(device)

            distractors = _batch['distractor_images']
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

            # for i in range(seq_length-1):
            i = 3
            imgs = references[i]
            refs = references[i+1]
                
            action = agent.policy(imgs)
            preds = torch.normal(action, args.variance)

            # reshape for concatenation
            preds = torch.unsqueeze(preds, dim=1)
            refs = torch.unsqueeze(refs, dim=1)
            candidates = torch.cat([refs, feat_distractors], dim=1)

            preds = torch.repeat_interleave(preds, num_distractors+1, dim=1)
            feat_diff = torch.norm(preds - candidates, p=2, dim=2)

            # dump into json
            # _batch["image_id"], _batch["distractor_image_ids"]
            for i in range(batch_size):
                data = {}
                data['epoch'] = epoch_id
                data['iter_id'] = _iter_id
                data['context_image_ids'] = _batch["image_id"][i].tolist()
                data['candidates'] = _batch["distractor_image_ids"][i].tolist()
                data['scores'] = feat_diff[i].cpu().numpy().tolist()
                full_data.append(data)

            min_indices = torch.argmin(feat_diff, dim=1).flatten()
            r3_indices = torch.argsort(feat_diff, dim=1)
            zero_pos = torch.argmin(r3_indices, dim=1).flatten()

            zeros = min_indices == 0
            r3 = zero_pos < 3
            correct += zeros.nonzero().shape[0]
            r3_correct += r3.nonzero().shape[0]
            total += batch_size
    if val:
        logging.info("[Val] [Epoch #: %f]\t [Accuracy: %f]\t [R3 Accuracy: %f]\n" % (epoch_id, correct/(batch_size),r3_correct/(batch_size)))
        path = './logger/val_' + args.name + '.json'
        with open(path, 'w') as outfile:
            json.dump(full_data, outfile)
    else:
        logging.info("[Test] [Epoch #: %f]\t [Accuracy: %f]\t [R3 Accuracy: %f]\n" % (epoch_id, correct/(batch_size),r3_correct/(batch_size)))
        path = './logger/test_' + args.name + '.json'
        with open(path, 'w') as outfile:
            json.dump(full_data, outfile)

if __name__ == "__main__":
    test_loop()