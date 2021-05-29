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
		if self.args.env in ["gridWorld", "gridWorld2", "lightWorld"]:
			for episode in range(self.args.episodes):
				F_t = 0
				
				for p in self.val_net.parameters():
					p.grad = torch.zeros_like(p.data)

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
					
					if self.args.trace_type == "etd":
						F_t = self.args.gamma * F_t + self.args.intrst
						scale_val = (1-beta) * self.args.intrst + beta * F_t

					elif self.args.trace_type == "etd_adaptive":
						F_t = self.args.gamma * F_t + (self.args.intrst * beta)
						scale_val = (1-beta) * (self.args.intrst * beta) + beta * F_t

					elif self.args.trace_type == "gated":
						scale_val = beta

					elif self.args.trace_type == "accumulating":
						scale_val = 1.0

					else:
						raise NotImplementedError

					scale_past = self.args.gamma * (1 - beta)

					for p in self.val_net.parameters():
						p.grad = p.grad * scale_past

					(c_val * scale_val).backward()

					for p in self.val_net.parameters():
						p.data = p.data + self.args.lr * curr_error.detach() * p.grad

				if (episode+1)%self.args.test_every==0:
					mse = test_cls.MSE(self.val_net)
					error_list.append(mse.item())

		else:
			raise NotImplementedError
		
		return self.val_net, error_list