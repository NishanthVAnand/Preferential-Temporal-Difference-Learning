import numpy as np
import random
from gym import core, spaces
from gym.envs.registration import register
from scipy import signal

class gridWorld2():
    def __init__(self, n=6, slippery=0, po=True, seed=0): # size in multiples of 4 for symmetry
        self.seed(seed)
        self.n = n
        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.n), spaces.Discrete(self.n)))
        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.slippery = slippery
        self.goal = (self.n-1, self.n-1)
        if po:
            all_states = [(i,j) for i in range(self.n) for j in range(self.n)]
            all_states.remove(self.goal)
            self.po_states = random.sample(all_states, self.n**2//2)
        else:
            self.po_states = []
        self.init_states = [(i,j) for i in range(self.n//2) for j in range(self.n//2)]
        self.feat = self.features(self.goal)
        self.goal_reward = 5
        self.reward = 0.0

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def features(self, state):
        if state in self.po_states:
            img = np.random.normal(0,1, size=(self.n, self.n))
        else:
            img = np.zeros((self.n, self.n))
            img[state[0], state[1]] = 1
        return img

    def generateP(self):
        P = np.zeros((self.action_space.n, self.n**2, self.n**2))
        for idx_i in range(self.n):
            for idx_j in range(self.n):
                curr_cell = (idx_i,idx_j)
                for c_a in range(self.action_space.n):
                    next_cell = tuple(curr_cell + self.directions[c_a])
                    if not self.observation_space.contains(next_cell):
                        next_cell = curr_cell
                    P[c_a][self.n*idx_i+idx_j][self.n*next_cell[0]+next_cell[1]] += 1-self.slippery

                    empty_cells = self.empty_around(curr_cell)
                    empty_cells.remove(next_cell)
                    for idx_cell in empty_cells:
                        P[c_a][self.n*idx_i+idx_j][self.n*idx_cell[0]+idx_cell[1]] += 1/3 * self.slippery

        for c_a in range(self.action_space.n):
            P[c_a][self.n*self.goal[0]+self.n-1] = 0

        return P

    def generateR(self):
        R = np.zeros((self.action_space.n, self.n**2))
        for idx_i in range(self.n):
            for idx_j in range(self.n):
                curr_cell = (idx_i,idx_j)
                for c_a in range(self.action_space.n):
                    next_cell = tuple(curr_cell + self.directions[c_a])
                    if not self.observation_space.contains(next_cell):
                        next_cell = curr_cell
                    n_state = next_cell
                    if n_state == self.goal:
                        R[c_a][self.n*idx_i+idx_j] += self.goal_reward * (1 - self.slippery)
                    else:
                        R[c_a][self.n*idx_i+idx_j] += self.reward * (1 - self.slippery)

                    empty_cells = self.empty_around(curr_cell)
                    empty_cells.remove(next_cell)
                    for idx_cell in empty_cells:
                        if idx_cell == self.goal:
                            R[c_a][self.n*idx_i+idx_j] += 1/3 * self.slippery * self.goal_reward
                        else:
                            R[c_a][self.n*idx_i+idx_j] += self.reward * (1 - self.slippery)

        for c_a in range(self.action_space.n):
            R[c_a][self.n*self.goal[0]+self.n-1] = 0
        return R

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if self.observation_space.contains(nextcell):
                avail.append(nextcell)
            else:
                avail.append(cell)
        return avail

    def reset(self):
        idx = np.random.choice(range(len(self.init_states)))
        state = self.init_states[idx]
        self.currentcell = state
        return self.features(state), state, not(int(state in self.po_states))

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        reward = 0
        if np.random.uniform() < self.slippery:
            empty_cells = self.empty_around(self.currentcell)
            empty_cells.remove(tuple(self.currentcell + self.directions[action]))
            nextcell = empty_cells[np.random.randint(len(empty_cells))]
        else:
            nextcell = tuple(self.currentcell + self.directions[action])

        if self.observation_space.contains(nextcell):
            self.currentcell = nextcell
        state = self.currentcell
        
        if state == self.goal:
          reward = self.goal_reward
        else:
          reward = self.reward
        done = state == self.goal
        return self.features(state), state, int(not(state in self.po_states)), reward, done, None