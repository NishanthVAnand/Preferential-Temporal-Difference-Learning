import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle

from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collector import dataCollector
from train import trainer
from test import test

parser = ArgumentParser(description="Parameters for the code - gated trace")
parser.add_argument('--trace_type', type=str, default="gated", help="gated or accumulating" )
parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")
parser.add_argument('--n', type=int, default=6, help="chain length")
parser.add_argument('--t_seeds', type=int, default=25, help="total seeds")
parser.add_argument('--env', type=str, default="gridWorld", help="Environment")
parser.add_argument('--intrst', type=float, default=0.01, help="interest of states while using etd")
parser.add_argument('--test_every', type=int, default=1, help="calculate MSE after # episode")
parser.add_argument('--episodes', type=int, default=30, help="number of episodes")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--log', type=int, default=1, help="compute the results")
parser.add_argument('--save', type=int, default=1, help="save the results")
parser.add_argument('--fo', action='store_true', help="type of env")
parser.add_argument('--train_feat', action='store_true', help="train features or FA")

args = parser.parse_args()

total_seeds = args.t_seeds
model_params = {8:(0.001, 200), 12:(0.0005, 100), 16:(0.0005, 50)}

def weights_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight)
		torch.nn.init.ones_(m.bias)

for seed in range(total_seeds):
	args.seed = seed
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if args.env == "gridWorld":
		from grid_world import gridWorld
		from networks import gridNet, linearNet

		env = gridWorld(n=args.n, po = not(args.fo))

		if args.train_feat:
			val_net = gridNet(n=args.n)
			val_net.to(device)

		else:
			feat_net = gridNet(n=args.n)
			path = "size_"+str(args.n)+"_lr_"+str(model_params[args.n][0])+"_epi_"+str(model_params[args.n][1])+".pt"
			feat_net.load_state_dict(torch.load(path))
			feat_net.eval()
			feat_net.to(device)

			val_net = linearNet()

	elif args.env == "lightWorld":
		from light_world import lightWorld
		from networks import lightNet, linearNet

		env = lightWorld(n=args.n)

		if args.train_feat:
			val_net = lightNet(n=args.n)
			val_net.to(device)

		else:
			feat_net = gridNet(n=args.n)
			path = "size_"+str(args.n)+"_lr_"+str(model_params[args.n][0])+"_epi_"+str(model_params[args.n][1])+".pt"
			feat_net.load_state_dict(torch.load(path))
			feat_net.eval()
			feat_net.to(device)
			
			val_net = linearNet()

	env.seed(args.seed)

	data_collector = dataCollector(env, args)
	data_list = data_collector.collect_data()
	
	if args.train_feat:
		#import pdb; pdb.set_trace()
		val_net.apply(weights_init)
		optimizer = optim.Adam(val_net.parameters(), lr=args.lr)
		#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, threshold=0.03, verbose=True)
		#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
		scheduler = None
		train_cls = trainer(args, val_net, data_list, device, optimizer, scheduler)
	else:
		train_cls = trainer(args, feat_net, data_list, device, linearNet=val_net)
	
	test_cls = test(args, env, np.ones((env.n**2,env.action_space.n))/env.action_space.n, device)
	val_net, error_list = train_cls.train(test_cls)

	if args.save == 1:
		if args.train_feat:
			filename = "drl_MC_FO"+"_env_"+str(args.env)+"_size_"+str(args.n)+"_lr_"+str(args.lr)+"_seed_"+str(args.seed)+"_epi_"+str(args.episodes)

		else:
			if args.trace_type == "etd":
				filename = "drl_"+args.trace_type+"_int_"+str(args.intrst)+"_env_"+str(args.env)+"_size_"+str(args.n)+"_lr_"+str(args.lr)+"_seed_"+str(args.seed)+"_epi_"+str(args.episodes)
			else:
				filename = "drl_"+args.trace_type+"_env_"+str(args.env)+"_size_"+str(args.n)+"_lr_"+str(args.lr)+"_seed_"+str(args.seed)+"_epi_"+str(args.episodes)

		with open("results_"+str(args.env)+"/"+filename+"_all_errors.pkl", "wb") as f:
			pickle.dump(error_list, f)

	if args.log == 1 and args.train_feat:
		filename = "drl_MC_FO"+"_env_"+str(args.env)+"_size_"+str(args.n)+"_lr_"+str(args.lr)+"_seed_"+str(args.seed)+"_epi_"+str(args.episodes)+".pt"
		torch.save(val_net.state_dict(), "models_"+str(args.env)+"/"+filename)
