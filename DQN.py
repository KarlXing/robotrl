import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        self.f1 = nn.Linear(1536, 512)
        self.f2 = nn.Linear(512, 7)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.f1(x))
        x = self.f2(x)
        return x

Transition = namedtuple('Transition', 'state action next_state reward')

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self, model, target_model, lr, batch_size, grad_clamp, batch_upper = 64, gamma=0.99):
        self.model = model
        self.target_model = target_model
        self.lr = lr
        self.batch_size = batch_size
        self.batch_upper = batch_upper
        self.grad_clamp = grad_clamp
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, eps = 1e-5)

    def update(self, memory):
        if len(memory) < self.batch_size:
            return
        # batch_size = self.batch_upper if len(memory) > self.batch_upper else len(memory)
        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                    batch.next_state)), dtype = torch.uint8)
        # non_final_next_states = torch.from_numpy(np.transpose(np.asarray([s for s in batch.next_state if s is not None]),
        #                             (0,3,1,2))).float()
        non_final_next_states = torch.from_numpy(np.asarray([s for s in batch.next_state if s is not None])).float()
        state_batch = torch.from_numpy(np.asarray(batch.state)).float()
        reward_batch = torch.tensor(batch.reward)
        action_batch = torch.tensor([[s] for s in batch.action])
        state_action_values = self.model(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-self.grad_clamp, self.grad_clamp)
        self.optimizer.step()