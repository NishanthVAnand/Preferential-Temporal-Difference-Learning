import torch
import torch.nn as nn
import torch.nn.functional as F

class gridNet(nn.Module):
  def __init__(self, n):
    super(gridNet, self).__init__()
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
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

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
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x