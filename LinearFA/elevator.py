import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
from copy import deepcopy
import numpy as np

class elevator():
    def __init__(self, n=5, var=0.2):
        self.coridor_chain = n #length of one corridor
        self.var = var
        self.state = (0,3)  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2) #number of actions - 2 - [0: go up, 1: go down]
        self.elevator_states = [(0,3), (1+self.coridor_chain, 4), (1+self.coridor_chain, 2)]
        self.goal_states = [(2+2*self.coridor_chain,3), (2+2*self.coridor_chain,5), (2+2*self.coridor_chain,1)]
        self.goal_reward = {(2+2*self.coridor_chain,3):2, (2+2*self.coridor_chain,5):3, (2+2*self.coridor_chain,1):1}
        self.observation_space = spaces.Tuple((spaces.Discrete(3+2*self.coridor_chain), spaces.Discrete(5)))
        self.feat = self.features(self.state)

    def generate_rew(self, state):
        if state in self.goal_states:
            return np.random.normal(self.goal_reward[state], self.var)
        else:
            return 0

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def features(self, state):
        temp_feat = [0, 0, 0, 0, 0, 0]
        if state in self.goal_states:
            temp_feat[len(self.elevator_states)+self.goal_states.index(state)] = 1
        elif state in self.elevator_states:
            temp_feat[self.elevator_states.index(state)] = 1
        else:
            temp_feat = (0.5+np.random.randn(6)).tolist()
        return temp_feat

    def step(self, action):
        '''
        takes an action as an argument and returns the next_state, reward, done, info.
        '''
        assert self.action_space.contains(action)
        reward = 0
        done = False
        
        # give a reward if in the terminal state
        if self.state in self.goal_states:
                reward = self.generate_rew(self.state)
                done = True
                self.state = self.reset()

        else: 
            # elevator takes up or down
            if self.state in self.elevator_states:
                if not action:
                    self.state = (self.state[0], self.state[1] + 1)
                else:
                    self.state = (self.state[0], self.state[1] - 1)

            self.state = (self.state[0]+1, self.state[1]) # move forward along the chain every step

        return self.features(self.state), self.state, reward, done, {}

    def reset(self):
        '''
        transitions back to first state
        '''
        self.state = (0,3)
        return self.features(self.state), self.state
