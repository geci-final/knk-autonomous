import torch
from ._rl_models import Actor, DoubleCritic
from .rl_utils import ReplayBuffer, AgentMode
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple


class TD3:
    def __init__(self, state_dim: int, action_dim: int, max_action_val: float, checkpt_file_path: str, log_dir: str, device: Optional[torch.device] = None, REPLAY_CAPACITY: int = int(1e6),
                 gamma: float = 0.99, tau: float = 0.005, actor_lr: float = 1e-4, critic_lr: float = 1e-3, policy_noise: float = 0.2, noise_clip: float = 0.5, policy_delay: int = 2,
                 load_checkpoint=False, mode=AgentMode.TRAIN):
        self.GAMMA = gamma
        self.TAU = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.max_action_val = max_action_val
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.iter_count = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.max_action_val).to(self.device)
        self.actor_target = Actor(
            self.state_dim, self.action_dim, self.max_action_val).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        self.critic = DoubleCritic(
            self.state_dim, self.action_dim).to(self.device)
        self.critic_target = DoubleCritic(
            self.state_dim, self.action_dim).to(self.device)
        if load_checkpoint and checkpt_file_path:
            self.load_checkpoint(checkpt_file_path)
        else:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_optmizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr)
        self.loss = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer(
            capacity=REPLAY_CAPACITY, state_dim=self.state_dim, action_dim=self.action_dim)
        self.agent_mode = mode
        self.writer = SummaryWriter(log_dir=log_dir)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        if self.agent_mode == AgentMode.TRAIN:
            action = self.actor(state)
            noise = torch.randn_like(action)*self.policy_noise
            action = torch.clamp(
                action+noise, -self.max_action_val, self.max_action_val)
            return action.detach().cpu()
        else:
            action = self.actor(state)
            return action.detach().cpu()

    def train(self, batch_size: int) -> Optional[Tuple[float, float]]:
        state, action, reward, next_state, done = self.replay_buffer.get_sample(
            batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        with torch.no_grad():
            noise = torch.randn_like(action)*self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state)+noise
            next_action = torch.clamp(
                next_action, -self.max_action_val, self.max_action_val)
            q_target1, q_target2 = self.critic_target(next_state, next_action)
            q_target = torch.min(q_target1, q_target2)
            q_target = reward + self.GAMMA * q_target * (1-done)

        q1, q2 = self.critic(state, action)
        critic_loss = self.loss(q1, q_target) + self.loss(q2, q_target)
        self.critic_optmizer.zero_grad()
        critic_loss.backward()
        self.critic_optmizer.step()
        self.iter_count += 1

        if self.iter_count % self.policy_delay == 0:

            for params in self.critic.parameters():
                params.requires_grad = False

            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for params in self.critic.parameters():
                params.requires_grad = True

            with torch.no_grad():
                for params, target_params in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_params.data.copy_(
                        self.TAU*params.data+(1-self.TAU)*target_params.data)
                for params, target_params in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_params.data.copy_(
                        self.TAU*params.data+(1-self.TAU)*target_params.data)
            return actor_loss.item(), critic_loss.item()

    def save_checkpoint(self, filename: str):
        print("Saving checkpoint")
        torch.save(self.actor.state_dict(), filename+"_actor.pth")
        torch.save(self.critic.state_dict(), filename+"_critic.pth")

    def load_checkpoint(self, filename: str):
        print("Loading checkpoint")
        self.actor.load_state_dict(torch.load(
            filename+"_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(
            filename+"_critic.pth", map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
