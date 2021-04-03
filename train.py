import numpy as numpy
import torch 
import torch.nn as nn 
import torchvision

from models.gail import Disciminator, Policy, Gail

# add variables here for parser later
seed = 7
n_updates = 100
n_iter = 100
batch_size = 64

# init environment

# init random seeding
np.random.seed(seed)
torch.manual_seed(seed)

def train_loop():
    # initialize env and expert trajectories

    # initialize models
    gail = Gail()

    # loop 
    for update in range(1, n_updates+1):
        # sample trajectory (potentially move some update code like sampling to here)
        gail.update()


