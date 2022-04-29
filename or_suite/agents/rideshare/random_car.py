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
        self.num_nodes = len(env_config['starting_state'])
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
        return

    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''

        # When travel time is not considered, taking a weighted random choice
        # based on number of cars available at each node
        if self.num_nodes + 2 == len(state):
            potential_actions = list(range(self.num_nodes))
            prob_weight = np.array(
                state[:self.num_nodes]) / (self.num_cars)
            action = np.random.choice(potential_actions, 1, p=prob_weight)[0]
            return action

        # When there are cars that are not in transit (state[-3] = # of cars in transit)
        if self.num_cars > state[-3]:
            potential_actions = list(range(self.num_nodes))
            prob_weight = np.array(
                state[:self.num_nodes]) / (self.num_cars - state[-3])
            action = np.random.choice(potential_actions, 1, p=prob_weight)[0]
        # When there aren't any cars available for dispatch
        else:
            action = np.random.choice(self.num_nodes)
        return action

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
