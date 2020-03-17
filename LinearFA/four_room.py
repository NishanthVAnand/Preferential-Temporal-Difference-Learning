import numpy as np
from gym import core, spaces
from gym.envs.registration import register
from scipy import signal

class Fourrooms():
    def __init__(self, slippery=1/3):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))
        self.hallway_states = [25, 51, 88]
        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.slippery = slippery
        self.kernel = np.random.normal(0, 1, 100)
        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62
        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)
        #self.init_states = [51]
        self.feat = self.features(self.goal)

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def features(self, state, kernel_size = (6,6)):
        img = np.zeros((11, 11))
        #kernel = np.geomspace(0.01,3,np.prod(kernel_size)).reshape(kernel_size)
        kernel = self.kernel[:np.prod(kernel_size)].reshape(kernel_size)
        cell = self.tocell[state]
        img[cell[0]-1, cell[1]-1] = 1
        out = signal.convolve2d(img, kernel, 'valid')
        return out.flatten().tolist()

    def generateP(self):
        P = np.zeros((self.action_space.n, len(self.tocell), len(self.tocell)))
        for c_state in range(len(self.tocell)):
            curr_cell = self.tocell[c_state]
            for c_a in range(self.action_space.n):
                next_cell = tuple(curr_cell + self.directions[c_a])
                if self.occupancy[next_cell]:
                    next_cell = curr_cell
                n_state = self.tostate[next_cell]
                P[c_a][c_state][n_state] += 1-self.slippery

                empty_cells = self.empty_around(curr_cell)
                empty_cells.remove(next_cell)
                for idx_cell in empty_cells:
                    P[c_a][c_state][self.tostate[idx_cell]] += 1/3 * self.slippery

        for c_a in range(self.action_space.n):
            P[c_a][self.goal] = 0

        return P

    def generateR(self):
        R = np.zeros((self.action_space.n, len(self.tocell)))
        for c_state in range(len(self.tocell)):
            curr_cell = self.tocell[c_state]
            for c_a in range(self.action_space.n):
                next_cell = tuple(curr_cell + self.directions[c_a])
                if self.occupancy[next_cell]:
                    next_cell = curr_cell
                n_state = self.tostate[next_cell]
                if n_state == self.goal:
                    R[c_a][c_state] += 1 * (1 - self.slippery)

                empty_cells = self.empty_around(curr_cell)
                empty_cells.remove(next_cell)
                for idx_cell in empty_cells:
                    idx_state = self.tostate[idx_cell]
                    if idx_state == self.goal:
                        R[c_a][c_state] += 1/3 * self.slippery * 1
        return R

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
            else:
                avail.append(cell)
        return avail

    def reset(self):
        state = np.random.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return self.features(state), state

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

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = state == self.goal
        return self.features(state), state, float(done), done, None
