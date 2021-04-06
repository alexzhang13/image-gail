import numpy as numpy
import torch 
import torch.nn as nn 
import torchvision

from models.gail import Disciminator, Policy, Gail
from models.expert import Expert

# add variables here for parser later
seed = 7
epochs = 100
n_updates = 100
batch_size = 16
learning_rate = 1e-4

# init random seeding
np.random.seed(seed)
torch.manual_seed(seed)


def train_loop():
    # initialize env and expert trajectories
    expert = Expert()

    # initialize models
    agent = Gail(lr=lr)
   

    # epoch loop
    for epoch in range(1, epochs+1):
        # rl loop for VIST dataset
        for n_update in n_updates:
            # sample trajectory (potentially move some update code like sampling to here)
            agent.update()


