import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

class YChain():
    def __init__(self, n=5):
        self.len_chain = n #length of one chain
        self.n = 1 + n*2 #length of MDP
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2) #number of actions - 2 - [0: go left, 1: go right]
        self.pos_reward = +2
        self.neg_reward = -1
        self.observation_space = spaces.Discrete(self.n)
        self.feat = self.features(self.state)
        self.bottle_neck = 0 #bottleneck state - where the transition happens

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def features(self, state):
        if state == 0:
            return [1, 0, 0]
        elif state == self.len_chain:
            return [0, 1, 0]
        elif state == self.len_chain*2:
            return [0, 0, 1]
        else:
            return (0.5+np.random.randn(3)).tolist()

    def step(self, action):
        '''
        takes an action as an argument and returns the next_state, reward, done, info.
        '''
        assert self.action_space.contains(action)
        reward = 0
        done = False
        
        # deciding on the next chain to switch if in the bottleneck state
        if self.state == self.bottle_neck:
            if not action:
                self.state += 1
            else:
                self.state = self.len_chain * 1 + 1
                
        # keep moving forward in the chain if not in the bottleneck state irrespective of the action
        else:
            
            # if in next transition is terminal state, give out reward
            if (self.state == self.len_chain * 1) or (self.state == self.len_chain * 2):
                reward = (self.pos_reward if self.state == self.len_chain*1 else self.neg_reward)
                done = True
                
            else:
                self.state += 1

        return self.features(self.state), self.state, reward, done, {}

    def reset(self):
        '''
        transitions back to first state
        '''
        self.state = 0
        return self.features(self.state), self.state
