import torch
import torch.nn as nn
import numpy as np

from grid_world import gridWorld
from light_world import lightWorld

class test():
	def __init__(self, args, env, policy, device):
		self.env = env
		self.args = args
		self.gamma = self.args.gamma
		self.P = self.env.generateP()
		self.R = self.env.generateR()
		self.policy = policy
		self.R_pi = np.einsum('as,sa->s', self.R, self.policy)
		self.P_pi = np.einsum('aij, ia -> ij', self.P, self.policy)
		self.v_pi = np.linalg.solve((np.eye(self.env.n**2) - self.gamma*self.P_pi), self.R_pi)
		self.criterion = nn.MSELoss()
		self.device = device

	def MSE_feat(self, val_net):
		test_data = []
		v_pi_mask = np.zeros((self.env.n**2,))

		for i in range(self.env.n**2):
			state = (i//self.env.n, i%self.env.n)

			if isinstance(self.env, gridWorld):
				if not (state in self.env.po_states):
					v_pi_mask[i] = 1
					temp = np.zeros((self.env.n,self.env.n))
					temp[i//self.env.n, i%self.env.n] = 1
					test_data.append(temp)

			elif isinstance(self.env, lightWorld):
				v_pi_mask[i] = 1
				temp = np.zeros((self.env.n,self.env.n))
				temp[i//self.env.n, i%self.env.n] = 1
				test_data.append(temp)
			else:
				raise NotImplementedError

		v_pred = []
		for t_data in test_data[:-1]:
  			c_f = torch.from_numpy(t_data).float().to(self.device)
  			with torch.no_grad():
  				_, pred = val_net(c_f.view(1,1,self.env.n,self.env.n))
  			v_pred.append(pred.detach().item())

		v_pi_m = self.v_pi * v_pi_mask
		v_pi_m = v_pi_m[v_pi_m != 0]

		return self.criterion(torch.tensor(v_pred), torch.from_numpy(v_pi_m).float())

	def MSE_linear(self, feat_net, val_net):
		test_data = []
		v_pi_mask = np.zeros((self.env.n**2,))

		for i in range(self.env.n**2):
			state = (i//self.env.n, i%self.env.n)

			if isinstance(self.env, gridWorld):
				if not (state in self.env.po_states):
					v_pi_mask[i] = 1
					temp = np.zeros((self.env.n,self.env.n))
					temp[i//self.env.n, i%self.env.n] = 1
					test_data.append(temp)

			elif isinstance(self.env, lightWorld):
				v_pi_mask[i] = 1
				temp = np.zeros((self.env.n,self.env.n))
				temp[i//self.env.n, i%self.env.n] = 1
				test_data.append(temp)
			else:
				raise NotImplementedError

		v_pred = []
		for t_data in test_data[:-1]:
  			c_f = torch.from_numpy(t_data).float().to(self.device)
  			with torch.no_grad():
  				c_f, _ = feat_net(c_f.view(1,1,self.env.n,self.env.n))
  			c_f = c_f.numpy().reshape(-1,1)
  			v_pred.append(val_net.forward(c_f).item())

		v_pi_m = self.v_pi * v_pi_mask
		v_pi_m = v_pi_m[v_pi_m != 0]

		return self.criterion(torch.tensor(v_pred).float(), torch.from_numpy(v_pi_m).float())

	def get_feat(self, val_net):
		test_data = []
		for i in range(self.env.n**2):
			state = (i//self.env.n, i%self.env.n)

			temp = np.zeros((self.env.n,self.env.n))
			temp[i//self.env.n, i%self.env.n] = 1
			test_data.append(temp)

		feat = []
		for t_data in test_data[:-1]:
  			c_f = torch.from_numpy(t_data).float().to(self.device)
  			with torch.no_grad():
  				pred, _ = val_net(c_f.view(1,1,self.env.n,self.env.n))
  			feat.append(pred.detach().numpy().to_list())

		return feat

