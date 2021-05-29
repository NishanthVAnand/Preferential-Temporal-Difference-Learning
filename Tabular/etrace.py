import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle
from argparse import ArgumentParser

parser = ArgumentParser(description="Parameters for the code - etrace")
parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--len', type=int, default=10, help="chain length")
parser.add_argument('--var', type=float, default=0.3, help="reward variance")
parser.add_argument('--env', type=str, default="simpleChain", help="Environment")
parser.add_argument('--episodes', type=int, default=30, help="number of episodes")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--lamb', type=float, default=0, help="lambda value")
parser.add_argument('--log', type=int, default=1, help="compute the results")
parser.add_argument('--save', type=int, default=1, help="save the results")
args = parser.parse_args()

if args.env == "randomWalk":

	def fbetas(env):
		def getbetas(state):
			return 1 - args.lamb
		return getbetas

	from random_walk import randomWalk

	env = randomWalk()
	betas = fbetas(env)
	action_1_prob = 0.5
	action_2_prob = 1-action_1_prob
	fo_states = [i for i in range(env.observation_space.n)]
	v_pi = {idx: i for idx, i in enumerate(np.arange(-18, 20, 2) / 20.0)}
	weights = 0.*np.ones_like(np.array(env.feat).reshape(-1,1))

else:
	raise NotImplementedError

env.seed(args.seed)
np.random.seed(args.seed)

def getAction(args):
	if args.env == "randomWalk":
		return np.random.binomial(1, action_1_prob)
	else:
		raise NotImplementedError

errors = []
emp_state_error = []

for n_epi in range(args.episodes):

	curr_s, c_s = env.reset() #curr_s: features, c_s: actual state
	trace = np.zeros_like(weights)
	done = False

	while(not done):
		next_s, n_s, reward, done, _ = env.step(getAction(args))
		curr_val = np.array(curr_s).T.dot(weights)
		if done:
			td_error = reward - curr_val
		else:
			next_val = np.array(next_s).T.dot(weights)
			td_error = reward + next_val - curr_val
		trace = np.array(curr_s).reshape(-1,1) + (1-betas(c_s)) * trace
		weights = weights + args.lr * td_error * trace 
		c_s = n_s
		curr_s = next_s

	if args.log == 1:
		value_pred = []
		value_true = []
		curr_error = 0
		for i_s in fo_states:
			feat = env.features(i_s)
			curr_value_pred = np.array(feat).T.dot(weights)
			value_pred.append(curr_value_pred.item())
			value_true.append(v_pi[i_s])

		sq_error = np.power(np.array(value_pred) - np.array(value_true), 2)
		curr_error = np.mean(sq_error)
		errors.append(curr_error)

if args.log == 1 and args.save == 1:
	filename = "etrace"+"_env_"+str(args.env)+"_lamb_"+str(args.lamb)+"_lr_"+str(args.lr)+"_seed_"+str(args.seed)

	with open("results_"+str(args.env)+"/"+filename+"_all_errors.pkl", "wb") as f:
		pickle.dump(errors, f)
