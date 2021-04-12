import numpy as np
import torch 
import torch.nn as nn 
import torchvision

from models.gail import Discriminator, Policy, Gail
from models.expert import Expert

# add variables here for parser later
seed = 7
epochs = 1
seq_length = 5
n_updates = 1
batch_size = 16
learning_rate = 1e-4
device = torch.device("cpu")

# init random seeding
np.random.seed(seed)
torch.manual_seed(seed)


def train_loop():
    # initialize env and expert trajectories
    expert = Expert()

    # initialize models
    agent = Gail(input_dim=(2*2048), lr=learning_rate, seq_length=seq_length, device=device)
   

    # epoch loop
    for epoch in range(1, epochs+1):
        # rl loop for VIST dataset
        for n_update in range(1,n_updates+1):
            print("Epoch #{}, Batch #{}".format(epoch, n_update))
            batch_raw = expert.sample(batch_size) # batch of images (fixed size) of length n
            batch_raw = np.reshape(batch_raw, (batch_size * seq_length, batch_raw.shape[2], batch_raw.shape[3], batch_raw.shape[4]))
            batch_raw = torch.FloatTensor(batch_raw)

            # sample trajectories
            exp_traj = agent.resnet(batch_raw)
            exp_traj = torch.reshape(exp_traj, (batch_size, seq_length, -1))

            state = exp_traj[:, 0] # get batch of first images of sequence
            sampled_traj = torch.unsqueeze(state, 1)
            for i in range(seq_length-1):
                action = agent.policy(state)
                sampled_traj = torch.cat((sampled_traj, torch.unsqueeze(torch.normal(action, 0.01), 1)), 1)
            
            agent.update(batch_size, sampled_traj, exp_traj)

    
    # apply some form of nearest neighbor here (?)
    print("Done!")

if __name__ == "__main__":
    train_loop()
