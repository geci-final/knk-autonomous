from typing import Tuple, Optional
import torch
from ._rl_models import Actor, Critic
from .rl_utils import OuNoise, ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


class DDPG:
    def __init__(self, state_dim: int, action_dim: int, max_action_val: float, checkpt_file_path: str, device: Optional[torch.device] = None, REPLAY_CAPACITY: int = int(1e5),
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3, weight_decay: float = 1e-2, gamma: float = 0.99, tau: float = 0.005, load_checkpoint=False, log_dir: str = "logs/ddpg"):
        self.GAMMA = gamma
        self.TAU = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action_val = max_action_val
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.max_action_val).to(self.device)
        self.actor_target = Actor(
            self.state_dim, self.action_dim, self.max_action_val).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, weight_decay=self.weight_decay)
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(
            self.state_dim, self.action_dim).to(self.device)
        if load_checkpoint and checkpt_file_path:
            self.load_checkpoint(checkpt_file_path)
        else:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_optmizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay)
        self.loss = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer(
            capacity=REPLAY_CAPACITY, state_dim=self.state_dim, action_dim=self.action_dim)
        self.OuNoise = OuNoise(size=self.action_dim, device=self.device)
        self.writer = SummaryWriter(log_dir=log_dir)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        action = self.actor(state)
        action += self.OuNoise.sample()
        return torch.clamp(action, -self.max_action_val, self.max_action_val).detach().cpu()

    def train(self, batch_size: int) -> Tuple[float, float]:
        state, action, reward, next_state, done = self.replay_buffer.get_sample(
            batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            q_target = reward + self.GAMMA * \
                self.critic_target(next_state, next_action) * (1-done)
        q_val = self.critic(state, action)
        critic_loss = self.loss(q_val, q_target)
        self.critic_optmizer.zero_grad()
        critic_loss.backward()
        self.critic_optmizer.step()
        for params in self.critic.parameters():
            params.requires_grad = False
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for params in self.critic.parameters():
            params.requires_grad = True
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.TAU*param.data + (1-self.TAU)*target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.TAU*param.data + (1-self.TAU)*target_param.data)
        return actor_loss.item(), critic_loss.item()

    def save_checkpoint(self, filename: str):
        print("Saving checkpoint")
        torch.save(self.actor.state_dict(), filename+"_actor.pth")
        torch.save(self.critic.state_dict(), filename+"_critic.pth")

    def load_checkpoint(self, filename: str):
        print("Loading checkpoint")
        self.actor.load_state_dict(torch.load(filename+"_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(filename+"_critic.pth", map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
