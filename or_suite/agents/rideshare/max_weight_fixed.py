import numpy as np
from .. import Agent


class maxWeightFixedAgent(Agent):
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

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.env_config = config
        return

    def update_obs(self, obs, action, reward, newObs, info):
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
        weighted_value = state / self.env_config['alpha']
        action = np.argmax(weighted_value, axis=1)

        return action

    def pick_action(self, state, step):
        action = self.greedy(state)
        return action
