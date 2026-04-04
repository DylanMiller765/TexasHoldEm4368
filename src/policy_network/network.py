import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, state_dim=324, action_dim=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(128, action_dim)

    def forward(self, x):
        logits = self.policy_head(self.net(x))
        return torch.softmax(logits, dim=-1)