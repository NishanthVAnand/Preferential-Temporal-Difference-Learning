import torch
import torch.nn as nn

import numpy as np

class trainer():
	def __init__(self, args, val_net, data_list, device, optimizer=None, scheduler=None, linearNet=None):
		self.args = args
		self.data_list = data_list
		self.optimizer = optimizer
		self.device = device
		self.val_net = val_net
		self.criterion = nn.MSELoss()
		self.scheduler = scheduler
		self.linearNet = linearNet
		torch.manual_seed(self.args.seed)

	def train(self, test_cls):
		if not(self.args.env == "gridWorld" or self.args.env=="gridWorld2"):
			print("Not a valid Environment")
			return 0

		error_list = []
		if self.args.train_feat:
			for episode in range(self.args.episodes):
				c_val = []
				c_tar = []
				rewards = []

				for data in self.data_list[episode]:
					c_f, _, rew, _, _ = data
					c_f = torch.from_numpy(c_f).float().to(self.device)
					_, value = self.val_net(c_f.view(1,1,self.args.n,self.args.n))
					c_val.append(value)
					rewards.append(rew)

				G = 0
				tar = []
				for r in reversed(rewards):
					G = self.args.gamma * G + r
					tar.append(G)

				error = self.criterion(torch.stack(c_val).view(-1,1), torch.tensor(tar[::-1]).to(self.device).view(-1,1))
				self.optimizer.zero_grad()
				error.backward()
				self.optimizer.step()
				if not(self.scheduler is None):
					self.scheduler.step()

				if (episode+1)%self.args.test_every==0:
					mse = test_cls.MSE_feat(self.val_net)
					error_list.append(mse.item())

			return self.val_net, error_list

		else:
			for episode in range(self.args.episodes):

				F_t = 0
				trace = np.zeros_like(self.linearNet.weights)

				for data in self.data_list[episode]:
					c_f, n_f, rew, beta, done = data
					beta = int(beta)
					c_f = torch.from_numpy(c_f).float().to(self.device)
					n_f = torch.from_numpy(n_f).float().to(self.device)
					
					with torch.no_grad():
						c_f, _ = self.val_net(c_f.view(1,1,self.args.n,self.args.n))
						n_f, _ = self.val_net(n_f.view(1,1,self.args.n,self.args.n))
					
					c_f = c_f.numpy().reshape(-1,1)
					n_f = n_f.numpy().reshape(-1,1)
					c_val = self.linearNet.forward(c_f)
					n_val = self.linearNet.forward(n_f)

					if not done:
						td_error = (rew + self.args.gamma * n_val - c_val)
					else:
						td_error = rew - c_val

					if self.args.trace_type == "etd":
						F_t = self.args.gamma * F_t + self.args.intrst
						M_t = (1-beta) * self.args.intrst + beta * F_t
						trace = self.args.gamma * (1-beta) * trace + M_t * c_f

					elif self.args.trace_type == "etd_adaptive":
						F_t = self.args.gamma * F_t + self.args.intrst * beta
						M_t = (1-beta) * self.args.intrst * beta + beta * F_t
						trace = self.args.gamma * (1-beta) * trace + M_t * c_f

					elif self.args.trace_type == "ptd":
						trace = self.args.gamma * (1-beta) * trace + beta * c_f

					elif self.args.trace_type == "accumulating":
						trace = self.args.gamma * (1-beta) * trace + c_f

					else:
						raise NotImplementedError

					self.linearNet.weights += self.args.lr * td_error * trace

				if (episode+1)%self.args.test_every==0:
					mse = test_cls.MSE_linear(self.val_net, self.linearNet)
					error_list.append(mse.item())
		
			return self.linearNet, error_list