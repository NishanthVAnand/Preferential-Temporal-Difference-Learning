import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

'''
class gridNet(nn.Module):
  def __init__(self, n):
    super(gridNet, self).__init__()
    self.n = n
    self.conv1 = nn.Conv2d(1, 8, 3)
    self.conv2 = nn.Conv2d(8, 16, 3)
    self.conv3 = nn.Conv2d(16, 16, 3)
    self.fc1 = nn.Linear(16*(self.n-6)*(self.n-6), 16)
    self.fc2 = nn.Linear(16, 1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 16*(self.n-6)*(self.n-6))
    feat = F.relu(self.fc1(x))
    x = self.fc2(feat)
    return feat, x

'''
class gridNet(nn.Module):
    def __init__(self, n):
        super(gridNet, self).__init__()
        self.n = n
        self.fc1 = nn.Linear(self.n*self.n, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        feat = F.relu(self.fc1(x.flatten()))
        x = self.fc3(feat)
        return feat, x
#'''

class lightNet(nn.Module):
  def __init__(self, n):
    super(lightNet, self).__init__()
    self.n = n
    self.conv1 = nn.Conv2d(1, 8, 3)
    self.conv2 = nn.Conv2d(8, 16, 3)
    self.conv3 = nn.Conv2d(16, 16, 3)
    self.fc1 = nn.Linear(16*(self.n-6)*(self.n-6), 64)
    self.fc2 = nn.Linear(64, 1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 16*(self.n-6)*(self.n-6))
    feat = F.relu(self.fc1(x))
    x = self.fc2(feat)
    return feat, x

class linearNet():
    def __init__(self, features=64):
        self.weights = np.zeros((features, 1))

    def forward(self, x):
        return self.weights.T.dot(x)