import numpy as np
import networkx as nx
import copy
from .. import Agent


class randomCarAgent(Agent):
    def __init__(self, epLen, env_config):
        """
        Args:
            epLen: number of steps
            func: function used to decide action
            env_config: parameters used in initialization of environment
            data: all data observed so far
        """
        self.env_config = env_config

        self.num_cars = env_config['num_cars']
        self.epLen = epLen
        self.data = []

    def reset(self):
        self.data = []

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.env_config = config
        return

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''
        self.data.append(newObs)
        return

    def update_policy(self, h):
        '''Update internal policy based upon records'''
        self.greedy = self.greedy
        return

    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''

        potential_actions = list(range(len(state[:-2])))
        prob_weight = np.array(state[:-2]) / self.num_cars
        action = np.random.choice(potential_actions, 1, p=prob_weight)[0]

        return action

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
