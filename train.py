import numpy as np
import torch 
import torch.nn as nn 
import torchvision

from models.gail import Discriminator, Policy, Gail
from models.expert import Expert

# add variables here for parser later
seed = 7
epochs = 1
n_updates = 15
batch_size = 16
learning_rate = 1e-4
device = torch.device("cuda")

# init random seeding
np.random.seed(seed)
torch.manual_seed(seed)


def train_loop():
    # initialize env and expert trajectories
    expert = Expert()

    # initialize models
    agent = Gail(input_dim=2048, lr=learning_rate, device=device)
   

    # epoch loop
    for epoch in range(1, epochs+1):
        # rl loop for VIST dataset
        for n_update in range(1,n_updates+1):
            print("Epoch #{}, Batch #{}".format(epoch, n_update))
            # sample trajectory (potentially move some update code like sampling to here)
            batch_raw = expert.sample(batch_size) # batch of images (fixed size) of length n
            agent.update(batch_raw, batch_raw)

    
    # apply some form of nearest neighbor here (?)

if __name__ == "__main__":
    train_loop()
