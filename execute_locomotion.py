import time
import datetime
import copy
import os, sys, shutil, argparse

import pickle
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from cem_policy.parallel_worker import ParallelRolloutWorker
import os
import pybullet as p
from cem_policy.utils import *

class CEMOptimizer(object):
    def __init__(self, cost_function, solution_dim, plan_n_segs, max_iters, population_size, num_elites,
                 upper_bound=None, lower_bound=None, epsilon=0.05):
        """
        :param cost_function: Takes input one or multiple data points in R^{sol_dim}\
        :param solution_dim: The dimensionality of the problem space
        :param max_iters: The maximum number of iterations to perform during optimization
        :param population_size: The number of candidate solutions to be sampled at every iteration
        :param num_elites: The number of top solutions that will be used to obtain the distribution
                            at the next iteration.
        :param upper_bound: An array of upper bounds for the sampled data points
        :param lower_bound: An array of lower bounds for the sampled data points
        :param epsilon: A minimum variance. If the maximum variance drops below epsilon, optimization is stopped.
        """
        super().__init__()
        self.solution_dim, self.max_iters, self.population_size, self.num_elites = \
            solution_dim, max_iters, population_size, num_elites
        self.plan_n_segs = plan_n_segs

        self.ub, self.lb = upper_bound.reshape([1, solution_dim]), lower_bound.reshape([1, solution_dim])
        self.epsilon = epsilon

        if num_elites > population_size:
            raise ValueError("Number of elites must be at most the population size.")

        self.cost_function = cost_function

    def obtain_solution(self, cur_state, init_mean=None, init_var=None):
        """ Optimizes the cost function using the provided initial candidate distribution
        :param cur_state: Full state of the current environment such that the environment can always be reset to this state
        :param init_mean: (np.ndarray) The mean of the initial candidate distribution.
        :param init_var: (np.ndarray) The variance of the initial candidate distribution.
        :return:
        """
        mean = (self.ub + self.lb) / 2. if init_mean is None else init_mean
        var = (self.ub - self.lb) / 4. if init_var is None else init_var
        t = 0
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))

        while (t < self.max_iters):  # and np.max(var) > self.epsilon:
            print("inside CEM, iteration {}".format(t))
            samples = X.rvs(size=[self.population_size, self.solution_dim]) * np.sqrt(var) + mean
            samples = np.clip(samples, self.lb, self.ub)
            full_samples = np.tile(samples, [1, self.plan_n_segs])
            costs_ = self.cost_function(cur_state, full_samples)
            costs = [_[0] for _ in costs_]
            print(np.mean(costs), np.min(costs))
            sort_costs = np.argsort(costs)

            elites = samples[sort_costs][:self.num_elites]
            mean = np.mean(elites, axis=0)
            var *= 0.2
            t += 1
        sol, solvar = mean, var
        sol = np.tile(sol, self.plan_n_segs)
        solvar = np.tile(solvar, [1, self.plan_n_segs])
        return sol


class CEMPolicy(object):
    """ Use the ground truth dynamics to optimize a trajectory of actions. """

    def __init__(self, env, env_class, env_kwargs, use_mpc, plan_horizon, plan_n_segs, max_iters, population_size, num_elites):
        self.env, self.env_class, self.env_kwargs = env, env_class, env_kwargs
        self.use_mpc = use_mpc
        self.plan_horizon, self.action_dim = plan_horizon, len(env.action_space.sample())
        self.plan_n_segs = plan_n_segs
        self.action_buffer = []
        self.prev_sol = None
        self.rollout_worker = ParallelRolloutWorker(env_class, env_kwargs, plan_horizon, self.action_dim)

        lower_bound = np.tile(env.action_space.low[None], [int(self.plan_horizon / self.plan_n_segs), 1]).flatten()
        upper_bound = np.tile(env.action_space.high[None], [int(self.plan_horizon / self.plan_n_segs), 1]).flatten()
        self.optimizer = CEMOptimizer(self.rollout_worker.cost_function,
                                      int(self.plan_horizon * self.action_dim / self.plan_n_segs),
                                      self.plan_n_segs,
                                      max_iters=max_iters,
                                      population_size=population_size,
                                      num_elites=num_elites,
                                      lower_bound=lower_bound,
                                      upper_bound=upper_bound, )

    def reset(self):
        self.prev_sol = None

    def get_action(self, state):
        if len(self.action_buffer) > 0 and not self.use_mpc:
            action, self.action_buffer = self.action_buffer[0], self.action_buffer[1:]
            return action

        env_state = save_env(self.env)

        soln = self.optimizer.obtain_solution(env_state, self.prev_sol).reshape([-1, self.action_dim])
        if self.use_mpc:
            self.prev_sol = np.vstack([np.copy(soln)[1:, :], np.zeros([1, self.action_dim])]).flatten()
        else:
            self.prev_sol = None
            self.action_buffer = soln[1:]  # self.action_buffer is only needed for the non-mpc case.
        load_env(self.env, state=env_state)  # Recover the environment
        print("cem finished planning!")
        return soln[0]

    

if __name__ == '__main__':
    import importlib
    import yaml

    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--env', default='open_the_dishwasher_door-v0',
                        help='Environment to train on (default: open_the_dishwasher_door-v0)')
    parser.add_argument('--algo', default='sac',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--task_config_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--render', type=int, default=0,
                        help='whether to use rendering (default: 0)')
    args = parser.parse_args()

    time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    robot_name = np.random.choice(['anymal', "a1"])

    horizon = 40
    config={
        'gui': args.render,
        'task': None,
        'robot_name': robot_name,
        'frameskip': 10,
        'frameskip_save': 2,
        'horizon': horizon,
    }

    default_cem_kwargs = {
        'use_mpc': False,
        'plan_horizon': horizon,
        'plan_n_segs': int(horizon/5),
        'max_iters': 5,
        'population_size': 6000,
        'num_elites': 1,
    }

    # change this to be the specified task class
    task_config = yaml.safe_load(open(args.task_config_path, 'r'))
    solution_path = task_config[0]['solution_path']
    task_name = solution_path.split("/")[-1][5:]
    module = importlib.import_module("{}.{}".format(solution_path.replace("/", "."), task_name))
    config["task_name"] = task_name
    env_class = getattr(module, task_name)
    env = env_class(**config)
    cem_config = copy.deepcopy(config)
    cem_config['gui'] = False

    policy = CEMPolicy(env,
                        env_class,
                        cem_config,
                        use_mpc=default_cem_kwargs['use_mpc'],
                        plan_horizon=default_cem_kwargs['plan_horizon'],
                        plan_n_segs=default_cem_kwargs['plan_n_segs'],
                        max_iters=default_cem_kwargs['max_iters'],
                        population_size=default_cem_kwargs['population_size'],
                        num_elites=default_cem_kwargs['num_elites'])
                            
    # Run policy

    all_rbgs = []
    all_states = []
    all_return = []

    obs = env.reset()

    rgbs = []
    states = []
    ret = 0
    done = False
    for idx in range(env.horizon):
        print("step {}".format(idx))
        action = policy.get_action(obs)
        obs, reward, done, _, rgbs_, states_ = env.step_(action)
        ret += reward
        rgbs += rgbs_
        states += states_

    save_path=f"{solution_path}/cem/{time_string}_{robot_name}_{ret:.3f}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_numpy_as_gif(np.array(rgbs), f"{save_path}/result.mp4", fps=60)
    pickle.dump(states, open(f"{save_path}/result.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)
    p.disconnect(env.id)
