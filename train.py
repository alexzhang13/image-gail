import numpy as np
import torch 
import torch.nn as nn 
import torchvision

from models.gail import Discriminator, Policy, Gail
from models.expert import Expert

torch.cuda.empty_cache()

# add variables here for parser later
seed = 7
epochs = 1
seq_length = 5
n_updates = 1
batch_size = 16
learning_rate = 1e-4
num_distractors = 4
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# init random seeding
np.random.seed(seed)
torch.manual_seed(seed)


def train_loop():
    # initialize env and expert trajectories
    freeze_resnet = True
    expert = Expert()

    # initialize models
    agent = Gail(input_dim=(2*2048), lr=learning_rate, seq_length=seq_length, device=device)
   
    # epoch loop
    for epoch in range(1, epochs+1):
        if epoch > 5 and freeze_resnet:
            agent.unfreeze_resnet()
            freeze_resnet = False

        # rl loop for VIST dataset
        for n_update in range(1,n_updates+1):
            print("Epoch #{}, Batch #{}".format(epoch, n_update))
            batch_raw = expert.sample(batch_size) # batch of images (fixed size) of length n
            batch_raw = np.reshape(batch_raw, (batch_size * seq_length, batch_raw.shape[2], batch_raw.shape[3], batch_raw.shape[4]))
            batch_raw = torch.FloatTensor(batch_raw).to(device)

            # sample trajectories
            exp_traj = agent.resnet(batch_raw)
            exp_traj = torch.reshape(exp_traj, (batch_size, seq_length, -1))

            state = exp_traj[:, 0] # get batch of first images of sequence
            sampled_traj = torch.unsqueeze(state, 1)
            for i in range(seq_length-1):
                action = agent.policy(state)
                sampled_traj = torch.cat((sampled_traj, torch.unsqueeze(torch.normal(action, 0.01), 1)), 1)
            
            agent.update(batch_size, sampled_traj, exp_traj)

        # maybe some form of validation?
        save_path = "./saved_models/checkpoint" + "_epoch_" + str(epoch) + ".t7"
        agent.save(save_path, epoch)
    
    # Evaluation
    expert_test = Expert()
    # for batch in test_dataloader:
    for i in range (1):
        batch_raw = expert.sample(batch_size) # batch of images (fixed size) of length n
        batch_raw = torch.FloatTensor(batch_raw).to(device)

        distractors = expert.sample_distractors(batch_size) # sample batch of distractors 
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
            candidates = torch.cat([preds, feat_distractors], dim=1)

            refs = torch.repeat_interleave(refs, num_distractors+1, dim=1)
            feat_diff = torch.norm(refs - candidates, p=2, dim=2)
            min_indices = torch.argmin(feat_diff, dim=1).flatten()
            zeros = min_indices == 0
            correct += zeros.nonzero().shape[0]
        
        accuracy = correct / (batch_size * (seq_length - 1))

    print("Accuracy: ", accuracy)
    # agent.save("./saved_models/", 1)

if __name__ == "__main__":
    train_loop()
