import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class randomWalk(gym.Env):
    def __init__(self, n=19):
        self.n = n
        self.state = 9  # Start at beginning of the chain
        self.reward = 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.feat = self.features(self.state)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def features(self, state):
        feat = [0]*self.n
        feat[state] = 1
        return feat

    def step(self, action):
        done = False
        reward = 0
        assert self.action_space.contains(action)

        if action:
            if self.state!=(self.n-1):
                self.state += 1

            else:
                done = True
                _, self.state = self.reset()
                reward = self.reward

        else:
            if self.state!=0:
                self.state -= 1

            else:
                done = True
                _, self.state = self.reset()
                reward = -1*self.reward

        return self.features(self.state), self.state, reward, done, {}

    def reset(self):
        self.state = 9
        return self.features(self.state), self.state