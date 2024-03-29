import numpy as np

class dataCollector():
  def __init__(self, env, args):
    self.env = env
    self.args = args
    np.random.seed(self.args.seed)

  def collect_data(self):
    data_list = []
    for epi in range(self.args.episodes):
      curr_epi_list = []
      curr_feat, curr_state, c_po = self.env.reset()
      done = False

      while not done:
        action = np.random.randint(0, self.env.action_space.n)
        next_feat, next_state, n_po, reward, done, _ = self.env.step(action)
        curr_epi_list.append((curr_feat, next_feat, reward, c_po, done))
        curr_feat = next_feat
        curr_state = next_state
        c_po = n_po

      data_list.append(curr_epi_list)

    return data_list

  def load_data(self):
    fname_data = "drl_env_"+str(self.args.env)+"_size_"+str(self.args.n)+"_seed_"+str(self.seed)+"_epi_250"
    with open("../NonLinearFA/data/"+fname_data+"_data.pkl", "rb") as f:
      data_list = pickle.load(f)

    return data_list