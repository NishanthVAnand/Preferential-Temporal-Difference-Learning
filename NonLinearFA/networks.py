import torch
import torch.nn as nn
import torch.nn.functional as F

class gridNet(nn.Module):
  def __init__(self, n):
    super(gridNet, self).__init__()
    self.n = n
    self.conv1 = nn.Conv2d(1, 16, 2)
    self.conv2 = nn.Conv2d(16, 16, 2)
    self.fc1 = nn.Linear(16*(self.n-2)*(self.n-2), 64)
    self.fc2 = nn.Linear(64, 1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(-1, 16*(self.n-2)*(self.n-2))
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x