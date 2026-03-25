import torch
import torch.nn as nn
import numpy as np 
import gymnasium as gym
import ale_py
from collections import deque
import random


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # 树的节点总数：对于有 capacity 个叶子节点的二叉树，内部节点有 capacity - 1 个
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority, data):
        # 找到叶子节点在 tree 数组中的索引
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # 覆盖旧数据
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        # 计算优先度的变化量
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # 向上回溯，更新所有父节点
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # 如果向下搜索到达了叶子节点
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # 决定向左还是向右走
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
                    
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


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


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # 决定优先级的程度，0 是均匀采样，1 是完全按 TD Error 采样
        self.beta = beta    # 用于 IS weight 修正偏差，初始值小于 1，慢慢增加到 1
        self.beta_increment = beta_increment
        self.epsilon = 1e-5 # 防止 TD Error 为 0 时优先级为 0，导致永远不被采样
        self.max_priority = 1.0 # 新进来的经验默认给最高优先级，保证至少被看过一次

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        
        # 将树的根节点（所有优先级的总和）分成 batch_size 个区间
        segment = self.tree.tree[0] / batch_size
        
        # 随着训练进行，beta 逐渐逼近 1
        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            
            idxs.append(idx)
            priorities.append(p)
            batch.append(data)

        # 计算 IS Weights (Importance Sampling Weights)
        sampling_probabilities = np.array(priorities) / self.tree.tree[0]
        is_weights = np.power(self.capacity * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max() # 归一化，保持更新稳定

        return batch, idxs, is_weights

    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            # 优先级 = (|TD Error| + epsilon) ^ alpha
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        # 实际存了多少数据，根据 pointer 计算
        if self.tree.tree[self.tree.capacity - 1] == 0:
            return self.tree.data_pointer
        else:
            return self.capacity


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
    def __init__(self, state_dim, action_dim, buffer_size=10000, batch_size=64, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.99):
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = Q_net(state_dim, action_dim).to(self.device)
        self.target_net = Q_net(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.prioritized_buffer = PrioritizedReplayBuffer(buffer_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad(): q_value = self.q_net(state)
            return q_value.argmax().item()
    
    def update(self):
        if len(self.prioritized_buffer) < self.batch_size:
            return
        # batch = self.buffer.sample(self.batch_size)
        batch, idxs, is_weights = self.prioritized_buffer.sample(self.batch_size)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        cur_q = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        td_errors = target_q - cur_q
        loss = (is_weights * (td_errors)**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.prioritized_buffer.update_priorities(idxs, td_errors.detach().cpu().numpy().squeeze())


def train():
    gym.register_envs(ale_py)
    env = gym.make('CartPole-v1')
    agent = DQN(env.observation_space.shape[0], env.action_space.n)

    for episode in range(10000):
        state, _ = env.reset()
        rewards = []

        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            agent.prioritized_buffer.add(state, action, reward, next_state, done)
            if len(agent.prioritized_buffer) >= agent.batch_size:
                agent.update()
            state = next_state
            rewards.append(reward)

            tau = 0.005
            for target_params, q_params in zip(agent.target_net.parameters(), agent.q_net.parameters()):
                target_params.data.copy_(tau * q_params.data + (1.0 - tau) * target_params.data)

            if done:
                break

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.001)

        if episode % 10 == 0:
            print(f'Episode {episode}, Reward: {sum(rewards)}')



    env.close()


if __name__ == "__main__":
    train()


