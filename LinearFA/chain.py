import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class simpleChain(gym.Env):
    def __init__(self, n=21, var=0.3):
        self.n = n
        self.state = 0  # Start at beginning of the chain
        self.mean = 1
        self.sigma = var
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(self.n-1)
        self.feat = self.features(self.state)
        self.seed()

    def features(self, state):
        return [1 * (self.n-state-1), 1]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        assert self.action_space.contains(action)
        
        if self.state!=(self.n-1):
            self.state+=1

            reward = np.random.normal(self.mean, self.sigma)

        if self.state == (self.n-1):
            done = True

        return self.features(self.state), self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.features(self.state), self.state