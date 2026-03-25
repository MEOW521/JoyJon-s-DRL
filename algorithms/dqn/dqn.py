import torch
import torch.nn as nn
import numpy as np 
import gymnasium as gym
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        assert len(self.buffer) >= batch_size, "Buffer size is less than batch size"
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Q_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state):
        return self.fc(state)


class DQN:
    def __init__(self, state_dim, action_dim, buffer_size=10000, batch_size=64, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.999):
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.q_net = Q_net(state_dim, action_dim)
        self.target_net = Q_net(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad(): q_value = self.q_net(state)
            return q_value.argmax().item()
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        cur_q = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(cur_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train():
    env = gym.make('CartPole-v1')
    agent = DQN(env.observation_space.shape[0], env.action_space.n)

    for episode in range(10000):
        state, _ = env.reset()
        rewards = []

        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            agent.buffer.add(state, action, reward, next_state, done)
            if len(agent.buffer) >= 1000:
                agent.update()
            state = next_state
            rewards.append(reward)

            if done:
                break

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

        if episode % 10 == 0:
            print(f'Episode {episode}, Reward: {sum(rewards)}')

        tau = 0.005
        for target_params, q_params in zip(agent.target_net.parameters(), agent.q_net.parameters()):
            target_params.data.copy_(tau * q_params.data + (1.0 - tau) * target_params.data)

    env.close()


if __name__ == "__main__":
    train()


