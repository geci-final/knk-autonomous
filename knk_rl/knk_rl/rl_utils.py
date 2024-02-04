import torch
from typing import Tuple
from enum import Enum


class AgentMode(Enum):
    TRAIN = 0
    EVAL = 1


class OuNoise:
    def __init__(self, size: int, device: torch.device, MU=0, THETA=0.15, SIGMA=0.2):
        self.size = size
        self.device = device
        self.MU = torch.full(
            (self.size,), MU, dtype=torch.float32).to(self.device)
        self.THETA = THETA
        self.SIGMA = SIGMA
        self.state = torch.ones(self.size, device=self.device)*self.MU

    def reset(self):
        self.state = torch.ones(self.size, device=self.device)*self.MU

    def sample(self) -> torch.Tensor:
        x = self.state
        dx = self.THETA*(self.MU-x)+self.SIGMA * \
            torch.randn(self.size, device=self.device)
        self.state = x+dx
        return self.state


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state = torch.empty((capacity, state_dim))
        self.action = torch.empty((capacity, action_dim))
        self.next_state = torch.empty((capacity, state_dim))
        self.reward = torch.empty((capacity, 1))
        self.done = torch.empty((capacity, 1))
        self.ptr = 0
        self.size = 0

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool):
        self.state[self.ptr] = state.detach().cpu()
        self.action[self.ptr] = action.detach().cpu()
        self.next_state[self.ptr] = next_state.detach().cpu()
        self.reward[self.ptr] = torch.tensor(
            reward, dtype=torch.float32).detach().cpu()
        self.done[self.ptr] = torch.tensor(
            done, dtype=torch.float32).detach().cpu()
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def get_sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,))
        return self.state[idx], self.action[idx], self.reward[idx], self.next_state[idx], self.done[idx]

    def __len__(self):
        return self.size
