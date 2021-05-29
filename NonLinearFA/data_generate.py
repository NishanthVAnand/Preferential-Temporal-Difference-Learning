import numpy as np
import gym
import pickle

from argparse import ArgumentParser

parser = ArgumentParser(description="Parameters for the code - gated trace")
parser.add_argument('--n', type=int, default=6, help="chain length")
parser.add_argument('--t_seeds', type=int, default=50, help="total seeds")
parser.add_argument('--env', type=str, default="gridWorld", help="Environment")
parser.add_argument('--episodes', type=int, default=250, help="number of episodes")
parser.add_argument('--fo', action='store_true', help="type of env")
args = parser.parse_args()

for seed in range(args.t_seeds):
	if args.env == "gridWorld":
		from grid_world import gridWorld
		env = gridWorld(n=args.n, po = not(args.fo))
		env.seed(seed)
		np.random.seed(seed)

	elif args.env == "gridWorld2":
		from grid_world2 import gridWorld2
		env = gridWorld2(n=args.n, po = not(args.fo))
		env.seed(seed)
		np.random.seed(seed)

	data_list = []
	for epi in range(args.episodes):
		curr_epi_list = []
		curr_feat, curr_state, c_po = env.reset()
		done = False

		while not done:
			action = np.random.randint(0, env.action_space.n)
			next_feat, next_state, n_po, reward, done, _ = env.step(action)

			curr_epi_list.append((curr_feat, next_feat, reward, c_po, done))

			curr_feat = next_feat
			curr_state = next_state
			c_po = n_po

		data_list.append(curr_epi_list)

	filename = "drl_env_"+str(args.env)+"_size_"+str(args.n)+"_seed_"+str(seed)+"_epi_"+str(args.episodes)
	with open("data/"+filename+"_data.pkl", "wb") as f:
		pickle.dump(data_list, f)