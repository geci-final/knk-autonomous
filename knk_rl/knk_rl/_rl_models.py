from typing import Tuple
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action_val: float, h1: int = 400, h2: int = 300):
        super(Actor, self).__init__()
        self.max_action_val = max_action_val
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        actions = self.max_action_val*torch.tanh(self.fc3(x))
        return actions


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, h1: int = 400, h2: int = 300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> float:
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        q_val = self.fc3(x)
        return q_val


class DoubleCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, h1: int = 400, h2: int = 300):
        super(DoubleCritic, self).__init__()
        # critic 1
        self.fc1 = nn.Linear(state_dim+action_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        # critic 2
        self.f4 = nn.Linear(state_dim+action_dim, h1)
        self.f5 = nn.Linear(h1, h2)
        self.f6 = nn.Linear(h2, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[float, float]:
        state_action = torch.cat([state, action], dim=1)
        x1 = torch.relu(self.fc1(state_action))
        x1 = torch.relu(self.fc2(x1))
        q1 = self.fc3(x1)

        x2 = torch.relu(self.f4(state_action))
        x2 = torch.relu(self.f5(x2))
        q2 = self.f6(x2)

        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> float:
        state_action = torch.cat([state, action], dim=1)
        x1 = torch.relu(self.fc1(state_action))
        x1 = torch.relu(self.fc2(x1))
        q1 = self.fc3(x1)
        return q1
