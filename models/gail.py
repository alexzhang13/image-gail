import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torch
import pdb

class Discriminator (nn.Module):
    def __init__(self, input_dim, device):
        super(Discriminator, self).__init__()
        self.device = device

        # F1024BDR - F512BDR - F1 - S
        self.fc1 = nn.Linear(in_features=input_dim,out_features=1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(in_features=1024,out_features=512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=1)

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

        l3 = self.fc3(l2)
        out = F.sigmoid(l3)
        return out



class Policy (nn.Module):
    def __init__(self, input_dim, device):
        super(Policy, self).__init__()
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
    def __init__(self, input_dim, lr, seq_length, device):
        super(Gail, self).__init__()
        self.lr = lr
        self.seq_length = seq_length
        self.device = device
        self.discount = 0.99

        # set up policy
        self.policy = Policy(2048, device)
        self.policy = nn.DataParallel(self.policy)
        self.policy.to(device)

        for param in self.policy.parameters():
            param.requires_grad = True
        
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler_policy = MultiStepLR(self.optim_policy, milestones=[20,40,60,80,100], gamma=0.1)

        # set up discriminator
        self.discriminator = Discriminator(input_dim, device)
        self.discriminator = nn.DataParallel(self.discriminator)
        self.discriminator.to(device)

        for param in self.discriminator.parameters():
            param.requires_grad = True

        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.scheduler_discriminator = MultiStepLR(self.optim_discriminator, milestones=[20,40,60,80,100], gamma=0.1)

        # deep representation 
        resnet = torchvision.models.resnet101(pretrained=True).to(device)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1]) 
        self.resnet = nn.DataParallel(self.resnet)
        self.resnet.to(device)

        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.optim_resnet = torch.optim.Adam(self.resnet.parameters(), lr=lr)
        self.scheduler_resnet = MultiStepLR(self.optim_resnet, milestones=[20,40,60,80,100], gamma=0.1)
  
        self.loss = nn.BCELoss().to(device)
        self.l1loss = nn.L1Loss().to(device)

        # cuda
        if torch.cuda.is_available():
            self.resnet.cuda()
            self.policy.cuda()
            self.discriminator.cuda()

    def on_epoch_end(self):
        self.scheduler_policy.step()
        self.scheduler_discriminator.step()
        self.scheduler_resnet.step()

    # ! imgs : B x 5 x 2048 (sampled trajectories policies)
    # ! imgs_expert: B x 5 x 2048(sampled trajectories from gt images)
    # ? discriminator loss: B x 5 x 2048 ---> (B X 4) x 4096 --> discriminator (cross entroy loss, label= 1 for imgs_expert and 0 for imgs)
    # ! sampling imgs --> take first image --> policy networks spits vector of size 2048 --> torch.normal(policy output, 0.01) --> use this to sample more hidden states until we get a sequence of size 5
    # ? for discriminator updates: detach on imgs, imgs_expert because we do not want to update \phi.
    # ? for policy updates: detach on policy prob, the reward from the discriminator is just a scalar value, don't want gradients to leak from there.
    # ? reinforce update : (Q in the supplementary: Q_t = mean(D(v_t, v_t+1) + D(v_t+1, v_t+2) + ..)).detach()
    # ? reinforce (state, rewards) --> rewards = Q.detach()
    def update(self, batch_size, sampled_states, exp_traj, log_probs, update_discrim):
        exp_state = exp_traj[:,0:self.seq_length-1]
        exp_action = exp_traj[:,1:self.seq_length]
        reshaped_exp_traj = torch.cat((exp_state, exp_action), axis=2)
        reshaped_exp_traj = torch.reshape(reshaped_exp_traj, (batch_size*(self.seq_length-1), -1))

        state = sampled_states[:,0:self.seq_length-1]
        action = sampled_states[:,1:self.seq_length]
        reshaped_samp_traj = torch.cat((state, action), axis=2)
        reshaped_samp_traj = torch.reshape(reshaped_samp_traj, (batch_size*(self.seq_length-1), -1))
        
        # label tensors and get policies
        policy_label = torch.zeros((batch_size*(self.seq_length-1),1), device=self.device)
        policy_prob = self.discriminator(reshaped_samp_traj.detach()) # detach phi output from discrim update

        # compute discrim loss
        if update_discrim:
            exp_label = torch.ones((batch_size*(self.seq_length-1),1), device=self.device)
            exp_prob = self.discriminator(reshaped_exp_traj.detach()) 

            self.optim_discriminator.zero_grad()
            
            discrim_loss = self.loss(exp_prob, exp_label) + self.loss(policy_prob, policy_label)
            discrim_loss_mean = discrim_loss.mean()
            discrim_loss_mean.backward()
            self.optim_discriminator.step()
        else:
            discrim_loss_mean = None

        # update policy: get loss from discrim using REINFORCE
        self.optim_policy.zero_grad()
        discrim_rewards = torch.reshape(policy_prob, (batch_size, (self.seq_length-1), -1))
        loss_policy = 0

        discount_factors = torch.Tensor([self.discount ** i for i in range(self.seq_length-1)]).to(self.device)

        for i in range (self.seq_length - 1):
            discount = discount_factors[:(self.seq_length - 1 - i)]
            Q = discrim_rewards[:,i:] * discount.unsqueeze(1)
            cur_reward = torch.sum(Q, dim=1)
            log_prob = log_probs[i]
            loss_policy += -1 * log_prob * (cur_reward.detach())
            

        # multiply by negative likelihood and q values
        loss_policy_mean = loss_policy.mean()
        loss_policy_mean.backward()
        self.optim_policy.step()

        return discrim_loss_mean, loss_policy_mean
        

    def unfreeze_resnet(self):
        for param in self.resnet.parameters():
            param.requires_grad = True

    def eval_mode(self):
        self.policy.eval()
        self.discriminator.eval()
        self.resnet.eval()

    def train_mode(self):
        self.policy.train()
        self.discriminator.train()
        self.resnet.train()

    def save(self, save_path, epoch):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'resnet_state_dict': self.resnet.state_dict(),
            'optimizer_policy_state_dict': self.optim_policy.state_dict(),
            'optimizer_discrim_state_dict': self.optim_discriminator.state_dict(),
            'optimizer_resnet_state_dict': self.optim_resnet.state_dict(),
            'epoch': epoch},
        save_path)

    def load(self, save_path):
        return torch.load(save_path)

    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.resnet.load_state_dict(checkpoint['resnet_state_dict'])

        self.optim_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
        self.optim_discriminator.load_state_dict(checkpoint['optimizer_discrim_state_dict'])
        self.optim_resnet.load_state_dict(checkpoint['optimizer_resnet_state_dict'])

        epoch = checkpoint['epoch']
    
        return epoch

