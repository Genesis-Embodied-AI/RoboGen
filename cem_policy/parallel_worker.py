import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import pybullet as p
from .utils import *

# env = None
def get_cost(args):
    cur_state, action_trajs, env_class, env_kwargs, worker_i = args
    # global env
    # if env is None:
    #     # Need to create the env inside the function such that the GPU buffer is associated with the child process and avoid any deadlock.
    #     # Use the global variable to access the child process-specific memory

    # import pdb; pdb.set_trace()
    print("env ", env_kwargs)
    env = env_class(**env_kwargs)
    env.reset()

    N = action_trajs.shape[0]
    costs = []
    for i in range(N):
        # print(worker_i, f'{i}/{N}')
        load_env(env, state=cur_state)
        ret = 0
        rewards = []
        for action in action_trajs[i, :]:
            _, reward, _, _ = env.step(action)
            ret += reward
            rewards.append(reward)
        costs.append([-ret, rewards])
        # print('get_cost {}: {}'.format(i, ret))
    p.disconnect(env.id)
    return costs


class ParallelRolloutWorker(object):
    """ Rollout a set of trajectory in parallel. """

    def __init__(self, env_class, env_kwargs, plan_horizon, action_dim, num_worker=32):
        self.num_worker = num_worker
        self.plan_horizon, self.action_dim = plan_horizon, action_dim
        self.env_class, self.env_kwargs = env_class, env_kwargs
        self.pool = Pool(processes=num_worker)

    def cost_function(self, init_state, action_trajs):
        action_trajs = action_trajs.reshape([-1, self.plan_horizon, self.action_dim])
        splitted_action_trajs = np.array_split(action_trajs, self.num_worker)
        ret = self.pool.map(get_cost, [(init_state, splitted_action_trajs[i], self.env_class, self.env_kwargs, i) for i in range(self.num_worker)])
        # ret = get_cost((init_state, action_trajs, self.env_class, self.env_kwargs))
        flat_costs = [item for sublist in ret for item in sublist]  # ret is indexed first by worker_num then traj_num
        return flat_costs


if __name__ == '__main__':
    # Can be used to benchmark the system
    from manipulation.sim import SimpleEnv
    import copy
    from RL.train_RL_api import default_config
    import pickle

    task_config = "gpt_4/data/parsed_configs_semantic_articulated/test_without_table.yaml"

    config = copy.deepcopy(default_config)
    config['config_path'] = task_config
    config['gui'] = False
    env = SimpleEnv(**config)
    env.reset()

    env_class = SimpleEnv
    env_kwargs = config
    initial_state = "manipulation/gpt_tasks/Load_Dishes_into_Dishwasher/12594/RL/open_the_dishwasher_door/best_final_state.pkl"
    with open(initial_state, 'rb') as f:
        initial_state = pickle.load(f)

    action_trajs = []
    for i in range(700):
        action = env.action_space.sample()
        action_trajs.append(action)
    action_trajs = np.array(action_trajs)
    rollout_worker = ParallelRolloutWorker(env_class, env_kwargs, 10, 7)
    cost = rollout_worker.cost_function(initial_state, action_trajs)
    print('cost:', cost)
