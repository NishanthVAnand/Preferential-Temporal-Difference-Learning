import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle
from argparse import ArgumentParser

parser = ArgumentParser(description="Parameters for the code - gated trace")
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

if args.env == "simpleChain":
	filename = "etrace"+"_env_"+str(args.env)+"_len_"+str(args.len)+"_lr_"+str(args.lr)+"_seed_"+str(args.seed)

	def fbetas(env):
		def getbetas(state):
			if state in [0, args.len]:
				return 1
			else:
				return 0.01
		return getbetas
	
	from chain import simpleChain

	env = simpleChain(n=args.len+1)
	betas = fbetas(env)
	fo_states = [i for i in range(0, args.len)]
	v_pi = {i:j for i, j in zip(fo_states,list(reversed(range(1,args.len+1))))}
	weights = np.zeros_like(np.array(env.feat).reshape(-1,1))

elif args.env == "YChain":
	filename = "etrace"+"_env_"+str(args.env)+"_len_"+str(args.len)+"_lr_"+str(args.lr)+"_seed_"+str(args.seed)

	def fbetas(env):
		def getbetas(state):
			if state in [0, args.len, args.len*2]:
				return 1
			else:
				return 0
		return getbetas

	from delayed_effect import YChain

	env = YChain(n=args.len)
	betas = fbetas(env)
	action_1_prob = 0.5
	action_2_prob = 1-action_1_prob
	fo_states = [0, args.len, 2*args.len]
	v_pi = {0:0.5, args.len:2, 2*args.len:-1}	
	weights = np.zeros_like(np.array(env.feat).reshape(-1,1))

elif args.env == "elevator":
	filename = "etrace"+"_env_"+str(args.env)+"_len_"+str(args.len)+"_lr_"+str(args.lr)+"_seed_"+str(args.seed)
	
	def fbetas(env):
		def getbetas(state):
			if state in env.goal_states or state in env.elevator_states:
				return 1
			else:
				return 0.01
		return getbetas

	from elevator import elevator

	env = elevator(n=args.len)
	betas = fbetas(env)
	action_1_prob = 0.5
	action_2_prob = 1-action_1_prob
	fo_states = [(0,3), (args.len+1, 2), (args.len+1, 4), (2*args.len+2, 5), (2*args.len+2, 3), (2*args.len+2, 1)]
	v_pi = {i:0.5*(1+i[1]) for i in fo_states} 
	weights = np.zeros_like(np.array(env.feat).reshape(-1,1))

elif args.env == "randomWalk":

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

elif args.env == "Fourrooms":
	filename = "etrace"+"_env_"+str(args.env)+"_lr_"+str(args.lr)+"_seed_"+str(args.seed)
	
	def fbetas(env):
		def getbetas(state):
			if state in env.hallway_states:
				return 1
			else:
				return 0
		return getbetas

	from four_room import Fourrooms

	env = Fourrooms(slippery=0)
	betas = fbetas(env)
	fo_states = env.hallway_states
	policy = np.ones((env.observation_space.n, env.action_space.n))/env.action_space.n
	R_pi, P_pi = np.einsum('as,sa->s', env.generateR(), policy), np.einsum('ijk, ji -> jk', env.generateP(), policy)
	v_pi = {idx:i for idx, i in enumerate(np.linalg.solve((np.eye(env.observation_space.n) - 1.0*P_pi), R_pi))}
	weights = np.zeros_like(np.array(env.feat).reshape(-1,1))

else:
	raise NotImplementedError

env.seed(args.seed)
np.random.seed(args.seed)

def getAction(args):
	if args.env == "simpleChain":
		return 0
	elif args.env == "YChain":
		return np.random.binomial(1, action_1_prob)
	elif args.env == "elevator":
		return np.random.binomial(1, action_1_prob)
	elif args.env == "randomWalk":
		return np.random.binomial(1, action_1_prob)
	elif args.env == "Fourrooms":
		return np.random.randint(4)
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
		weights = weights + args.lr * td_error * trace #learning rate: 0.01
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

	with open("results_"+str(args.env)+"/"+filename+"_all_errors.pkl", "wb") as f:
		pickle.dump(errors, f)

# print(weights)
# print(value_pred, v_pi)
# fig, ax = plt.subplots(2,1)
# ax[0].plot(errors)
# ax[1].plot(emp_state_error)
# plt.show()
#plt.plot(errors)
#plt.show()
#print(np.mean([i**0.5 for i in errors]))