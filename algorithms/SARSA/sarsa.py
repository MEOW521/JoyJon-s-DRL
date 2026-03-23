import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


class Q_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    
    def forward(self, state):
        return self.fc(state)


class SARSA:
    def __init__(self, state_dim, action_dim, buffer_size=32, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.q_net = Q_net(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer = []
        self.buffer_size = buffer_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.q_net(state)
            return q_value.argmax().item()

    def update(self, state, action, reward, next_state, next_action, done):
        self.buffer.append((state, action, reward, next_state, next_action, done))
        if len(self.buffer) >= self.buffer_size:
            states, actions, rewards, next_states, next_actions, dones = zip(*self.buffer)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            next_actions = torch.LongTensor(next_actions).unsqueeze(1).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            cur_q = self.q_net(states).gather(1, actions)

            with torch.no_grad():
                tar_q = rewards + self.gamma * (1 - dones) * self.q_net(next_states).gather(1, next_actions)

            loss = nn.MSELoss()(cur_q, tar_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.buffer.clear()

            self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)


def train():
    env = gym.make('CartPole-v1')
    agent = SARSA(env.observation_space.shape[0], env.action_space.n, buffer_size=32, lr=1e-3, gamma=0.99, epsilon=0.1)

    for episode in range(10000):
        state, _ = env.reset()
        action = agent.select_action(state)
        rewards = []

        for t in range(10000):
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            rewards.append(reward)
            
            if done:
                break

        if episode % 10 == 0:
            print(f'Episode {episode}, Reward: {sum(rewards)}')
    env.close()


if __name__ == "__main__":
    train()