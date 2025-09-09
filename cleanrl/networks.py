import torch
import torch.nn as nn
import torch.nn.functional as F

class HighLevelController(nn.Module):
    def __init__(self, obs_dim, num_objectives, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + num_objectives, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_objectives)  # output weights for each objective

    def forward(self, obs, collected_rewards):
        # obs: env observation (flattened)
        # collected_rewards: cumulative rewards for each objective
        x = torch.cat([obs, collected_rewards], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        weights = F.softmax(self.out(x), dim=-1)  # preference weights
        return weights
