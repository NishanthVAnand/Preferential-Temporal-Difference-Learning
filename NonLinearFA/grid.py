import numpy as np
from gym import core, spaces
from gym.envs.registration import register
from scipy import signal

class gridWorld():
    def __init__(self, n=5, conv_size=3):
        self.n = n
        self.conv_size = conv_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.n), spaces.Discrete(self.n)))
        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.goal = [(0, 0), (0, self.n-1), (self.n-1, 0), (self.n-1, self.n-1)]
        self.start = ((self.n-1)//2, (self.n-1)//2)
        self.feat = self.features(self.start)
        self.goal_reward = {(0, 0):1, (0, self.n-1):-1, (self.n-1, 0):-1, (self.n-1, self.n-1):1}
        self.reward = 0.0

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def features(self, state):
        location = np.zeros((self.n, self.n))
        location[state] = 1
        conv_filter = np.ones((self.conv_size, self.conv_size))
        features = signal.convolve2d(location, conv_filter, "valid")
        return features.flatten()

    def reset(self):
        self.currentcell = self.start
        return self.start, self.features(self.start)

    def step(self, action):
        reward = 0
        done = 0

        nextcell = tuple(self.currentcell + self.directions[action])
        if self.observation_space.contains(nextcell):
            self.currentcell = nextcell
        state = self.currentcell
        
        if state in self.goal:
            reward = self.goal_reward[state]
            done = 1
        else:
            reward = self.reward

        return state, self.features(state), reward, done, None