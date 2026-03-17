import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym

class Policy(nn.Module):
    """Policy network for REINFORCE algorithm

    Args:
        state_dim: dimension of state
        action_dim: dimension of action
    """
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            # nn.Softmax(dim=-1) 这个是用来将输出转换为概率的，但是这里我们不需要，因为我们要用log_prob来计算损失
        )

    def forward(self, state):
        return self.fc(state)

class REINFORCE:

    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.policy = Policy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def select_action(self, state):
        x = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        action_probs = self.policy(x)
        action_dist = Categorical(logits=action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob, action_dist.entropy()

    def update(self, rewards, log_probs, entropies, entropy_coef=0.001):
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        R = 0
        returns = torch.zeros_like(rewards)

        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R

        # returns = torch.tensor(returns).float().to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 这里注意log_probs是[n, 1]，而returns是[n, ]，直接相乘会变成矩阵
        policy_loss = -(log_probs.view(-1) * returns).sum()

        # 考虑熵正则化
        entropy_loss = -entropies.sum()

        total_loss = policy_loss + entropy_coef * entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


def train():
    env = gym.make("CartPole-v1")
    agent = REINFORCE(env.observation_space.shape[0], env.action_space.n)

    for episode in range(10000):
        state, _ = env.reset()
        log_probs = []
        entropies = []
        rewards = []

        for t in range(1000):
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            state = next_state

            if done or truncated:
                break
        agent.update(rewards, log_probs, entropies)

        if episode % 10 == 0:
            print(f'Episode {episode}, Reward: {sum(rewards)}')

    env.close()


if __name__ == "__main__":
    train()
        
        



