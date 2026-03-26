import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
from torch.distributions import Categorical

class Actor_Critic_Net(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(Actor_Critic_Net, self).__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            out = self.shared_conv(dummy_input)
        flattened_size = out.shape[1]

        self.shared_fc = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
        )

        self.actor_fc = nn.Linear(512, action_dim)
        self.critic_fc = nn.Linear(512, 1)

    def forward(self, state):
        state = state.float() / 255.0
        conv_out = self.shared_conv(state)
        fc_out = self.shared_fc(conv_out)
        return self.actor_fc(fc_out), self.critic_fc(fc_out)


class A2C:
    def __init__(self, input_shape, action_dim, lr=1e-4, gamma=0.99, gae_lambda=0.95, entropy_coef=0.001, value_coef=0.5):
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef


        self.actor_critic_net = Actor_Critic_Net(input_shape, action_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic_net.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic_net.to(self.device)

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []


    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, values = self.actor_critic_net(state)
        action_dist = Categorical(logits=action_probs.squeeze(0))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        self.log_probs.append(log_prob)
        self.values.append(values)
        self.entropies.append(action_dist.entropy())

        return action.item()

    def store_transition(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def calc_adv_gae(self, next_values):
        adv = []
        gae = 0

        values_extend = self.values + [next_values]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values_extend[t+1] * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - self.dones[t])
            adv.insert(0, gae)
        
        adv = torch.tensor(adv, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.values, dtype=torch.float32).to(self.device)

        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns


    def update(self, next_state, done):
        if done:
            next_value = 0
        else:
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            if next_state.shape[-1] == 4:
                next_state = next_state.permute(0, 3, 1, 2)
            with torch.no_grad():
                _, value = self.actor_critic_net(next_state)
            next_value = value.item()
        
        adv, returns = self.calc_adv_gae(next_value)

        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze(-1)
        entropies = torch.stack(self.entropies)

        actor_loss = -(log_probs * adv).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.values.clear()
        self.entropies.clear()
        self.rewards.clear()
        self.dones.clear()


def make_atari_env(env_name):
    gym.register_envs(ale_py)
    env = gym.make(env_name, frameskip=1)

    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
    )

    env = FrameStackObservation(env, 4)
    return env


def train():
    env_name = "ALE/Pong-v5"
    env = make_atari_env(env_name)
    
    agent = A2C(env.observation_space.shape, env.action_space.n)

    update_steps = 256

    for episode in range(10000):
        state, _ = env.reset()
        rewards = []
        step_count = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            agent.store_transition(reward, done)
            state = next_state
            rewards.append(reward)
            step_count += 1

            if step_count % update_steps == 0 or done:
                agent.update(next_state, done)
            if done : break
        print(f'Episode {episode}, Reward: {sum(rewards)}')

    env.close()


if __name__ == "__main__":
    train()
        


