import torch.nn as nn
import torch

class Disciminator (nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(),
            nn.BatchNorm2d(),
            nn.Dropout(),
            nn.Relu(),
        ).to(device)

class Policy (nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()

class Gail (nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        self.policy = Policy(input_dim, hidden_dim, device)
        self.optim_policy = torch.optim.Adam()

        self.discriminator = Disciminator(input_dim, hidden_dim, device)
        self.optim_discriminator = torch.optim.Adam()

    def update():
        # sample trajectories
        exp_state, exp_action = expert.sample()
        exp_state = torch.FloatTensor(exp_state).to(device)
        exp_action = torch.FloatTensor(exp_action).to(device)

        state,_ = expert.sample() # get same states
        action = self.policy(state)

        # update discriminator: gan loss
        self.optim_discriminator.zero_grad()

        

        

        # update policy: get loss from discrim
        self.optim_policy.zero_grad()
        loss_policy = -self.discriminator(state, action)
        loss_policy.mean().backward()
        self.optim_policy.step()

