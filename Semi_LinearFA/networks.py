import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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

class linearNet():
    def __init__(self, features=32):
        self.weights = np.zeros((features, 1))

    def forward(self, x):
        return self.weights.T.dot(x)