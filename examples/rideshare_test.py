has_travel_time = False
algo_tune_on = False

import or_suite
import numpy as np
import itertools as it

import copy

import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd


import gym
import networkx as nx

CONFIG =  or_suite.envs.env_configs.rideshare_graph_default_config
CONFIG['epLen'] = 2
epLen = CONFIG['epLen']
nEps = 2
numIters = 25

DEFAULT_SETTINGS = {'seed': 1, 
                    'recFreq': 1, 
                    'dirPath': '../data/rideshare/', 
                    'deBug': False, 
                    'nEps': nEps, 
                    'numIters': numIters, 
                    'saveTrajectory': True, 
                    'epLen' : epLen,
                    'render': False,
                    'pickle': False
                    }

starting_state = CONFIG['starting_state']
num_cars = CONFIG['num_cars']
num_nodes = len(starting_state)

if has_travel_time:
    rideshare_env = gym.make('Rideshare-v1', config=CONFIG)
else:
    rideshare_env = gym.make('Rideshare-v0', config=CONFIG)
mon_env = Monitor(rideshare_env)

scaling_list = [0.1, 0.3, 1, 5]
observation_space = rideshare_env.observation_space
action_space = rideshare_env.action_space

agents = { #'SB PPO': PPO(MlpPolicy, mon_env, gamma=1, verbose=0, n_steps=epLen),
#'Random': or_suite.agents.rl.random.randomAgent(),
# 'maxweightfixed' : or_suite.agents.rideshare.max_weight_fixed.maxWeightFixedAgent(CONFIG['epLen'], CONFIG, [1 for _ in range(num_nodes)]),
# 'randomcar' : or_suite.agents.rideshare.random_car.randomCarAgent(CONFIG['epLen'], CONFIG),
'discreteql' : or_suite.agents.rl.discrete_ql.DiscreteQl(action_space, observation_space, epLen, scaling_list[0])
}

#param_list = [list(p) for p in it.product(np.linspace(0,1,4),repeat = len(starting_state))]

path_list_line = []
algo_list_line = []
path_list_radar = []
algo_list_radar= []

linspace_alpha = []

for agent in agents:
    print(agent)
    DEFAULT_SETTINGS['dirPath'] = '../data/rideshare_'+str(agent)+'_'+str(num_cars)
    if algo_tune_on and agent == 'maxweightfixed':
        or_suite.utils.run_single_algo_tune(rideshare_env,agents[agent], param_list, DEFAULT_SETTINGS)
    if agent == 'SB PPO':
        or_suite.utils.run_single_sb_algo(mon_env, agents[agent], DEFAULT_SETTINGS)
    else:
        or_suite.utils.run_single_algo(rideshare_env, agents[agent], DEFAULT_SETTINGS)

    path_list_line.append('../data/rideshare_'+str(agent)+'_'+str(num_cars))
    algo_list_line.append(str(agent))
    if agent != 'SB PPO':
        path_list_radar.append('../data/rideshare_'+str(agent)+'_'+str(num_cars))
        algo_list_radar.append(str(agent))
        