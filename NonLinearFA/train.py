import torch

class trainer():
	def __init__(self, args, data_list, val_net, optimizer, device):
		self.args = args
		self.data_list = data_list
		self.val_net = val_net
		self.optimizer = optimizer
		self.device = device
		torch.manual_seed(self.args.seed)

	def train(self, test_cls):
		error_list = []
		if self.args.env == "gridWorld":
			for episode in range(self.args.episodes):
				td_errors = []
				c_grad = []
				beta_list = []

				for data in self.data_list[episode]:
					c_f, n_f, rew, beta, done = data
					c_f = torch.from_numpy(c_f).float().to(self.device)
					n_f = torch.from_numpy(n_f).float().to(self.device)
					c_val = self.val_net(c_f.view(1,1,self.args.n,self.args.n))
					n_val = self.val_net(n_f.view(1,1,self.args.n,self.args.n))
					if not done:
						curr_error = (rew + self.args.gamma * n_val - c_val)
					else:
						curr_error = rew - c_val
					td_errors.append(curr_error.detach()[0])
					c_grad.append(c_val)
					beta_list.append(beta)

				if self.args.trace_type == "gated":
					t_n_bar = td_errors[-1]
					t_n = beta_list[-1] * c_grad[-1] * t_n_bar
					nxt_beta = beta_list[-1]
					gradient = -t_n
					for idx, (grad, delta, beta) in enumerate(zip(reversed(c_grad[:-1]), reversed(td_errors[:-1]), reversed(beta_list[:-1]))):
						t_n_bar = delta + self.args.gamma * (1-nxt_beta) * t_n_bar
						t_n = grad * beta * t_n_bar
						gradient -= t_n
						nxt_beta = beta

				elif self.args.trace_type == "accumulating":
					t_n_bar = td_errors[-1]
					t_n = c_grad[-1] * t_n_bar
					nxt_beta = beta_list[-1]
					gradient = -t_n
					for idx, (grad, delta, beta) in enumerate(zip(reversed(c_grad[:-1]), reversed(td_errors[:-1]), reversed(beta_list[:-1]))):
						t_n_bar = delta + self.args.gamma * (1-nxt_beta) * t_n_bar
						t_n = grad * t_n_bar
						gradient -= t_n
						nxt_beta = beta

				else:
					raise NotImplementedError

				self.optimizer.zero_grad()
				gradient.backward()
				self.optimizer.step()

				if (episode+1)%self.args.test_every==0:
					mse = test_cls.MSE(self.val_net)
					error_list.append(mse.item())

		elif self.args.env == "lightWorld":
			pass

		else:
			raise NotImplementedError
		
		return self.val_net, error_list