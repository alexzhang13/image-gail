import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

from models.expert import Expert

class Disciminator (nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()
        self.device = device

        # F1024BDR - F512BDR - F1 - S
        self.fc1 = nn.Linear(in_features=input_dim,out_features=1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(in_features=1024,out_features=512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=1)

    def forward(self, state_future, state):
        # fbdr1024
        l1 = self.fc1(state)
        l1 = self.bn1(l1)
        l1 = F.dropout(l1)
        l1 = F.relu(l1)

        # fbdr512
        l2 = self.fc2(l1)
        l2 = self.bn2(l2)
        l2 = F.dropout(l2)
        l2 = F.relu(l2)

        l3 = self.fc3(l2)
        out = F.softmax(l3)
        return out



class Policy (nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.device = device

        self.fc1 = nn.Linear(in_features=input_dim,out_features=1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(in_features=1024,out_features=512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(in_features=512,out_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.fc4 = nn.Linear(in_features=256,out_features=512)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.fc5 = nn.Linear(in_features=512,out_features=1024)
        self.bn5 = nn.BatchNorm1d(num_features=1024)
        self.fc6 = nn.Linear(in_features=1024,out_features=2048)

    def forward(self, state):
        # fbdr1024
        l1 = self.fc1(state)
        l1 = self.bn1(l1)
        l1 = F.dropout(l1)
        l1 = F.relu(l1)

        # fbdr512
        l2 = self.fc2(l1)
        l2 = self.bn2(l2)
        l2 = F.dropout(l2)
        l2 = F.relu(l2)

        # fbdr256
        l3 = self.fc3(l2)
        l3 = self.bn3(l3)
        l3 = F.dropout(l3)
        l3 = F.relu(l3)

        # fbdr512
        l4 = self.fc4(l3)
        l4 = self.bn4(l4)
        l4 = F.dropout(l4)
        l4 = F.relu(l4)

        #fbdr1024
        l5 = self.fc5(l4)
        l5 = self.bn5(l5)
        l5 = F.dropout(l5)
        l5 = F.relu(l5)

        out = self.fc6(l5)
        return out

class Gail (nn.Module):
    def __init__(self, input_dim, hidden_dim, lr, device):
        # add learning rate
        self.policy = Policy(input_dim, hidden_dim, device)
        self.optim_policy = torch.optim.Adam(lr=lr)

        self.discriminator = Disciminator(input_dim, hidden_dim, device)
        self.optim_discriminator = torch.optim.Adam(lr=lr)

        # deep representation 
        self.resnet = torchvision.models.resnet101(pretrained=true) # not sure if this is pretrained
  
        self.expert = Expert()
        self.loss = nn.BCELoss()

    def update(self, batch_size=16, freeze_resnet=True):
        # sample trajectories
        exp_state, exp_action = expert.sample(batch_size)
        exp_state = torch.FloatTensor(exp_state).to(device)
        exp_action = torch.FloatTensor(exp_action).to(device)

        state,_ = expert.sample(batch_size) # get same state
        state = torch.FloatTensor(state).to(device)
        action = self.policy(state)

        # update discriminator: discrim loss
        self.optim_discriminator.zero_grad()
        
        # label tensors and get policies
        exp_label = torch.ones((batch_size,1), device=device)
        policy_label = torch.zeros((batch_size,1), device=device)

        exp_prob = self.discriminator(exp_state, exp_action) 
        policy_prob = self.discriminator(state, action.detach()) # detach policy output from discrim update

        # compute discim loss
        discrim_loss = self.loss(exp_prob, exp_label) + self.loss(policy_prob, policy_label)
        discrim_loss.backward()
        self.optim_discriminator.step()
        

        # update policy: get loss from discrim (wrong for now, should be policy gradient w roll out)
        # TODO: change update function
        self.optim_policy.zero_grad()
        loss_policy = -self.discriminator(state, action).detach() # detach discrim output from policy update
        loss_policy.mean().backward()
        self.optim_policy.step()

    def save(self, directory):
        pass

    def load(self, directoy):
        pass

