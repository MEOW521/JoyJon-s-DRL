import torch
import torch.nn as nn


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
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.001):
        self.q_net = Q_net(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = 


