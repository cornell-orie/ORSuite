import numpy as np
from .. import Agent


class maxWeightFixedAgent(Agent):
    def __init__(self, epLen, num_cars, alpha):
        """
        Args:
            epLen: number of steps
            func: function used to decide action
            env_config: parameters used in initialization of environment
            data: all data observed so far
        """
        self.data = []
        self.epLen = epLen
        self.num_cars = num_cars
        self.alpha = alpha

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''
        self.data.append(newObs)
        return

    def reset(self):
        self.data = []

    def update_policy(self, h):
        '''Update internal policy based upon records'''
        self.greedy = self.greedy
        return

    def greedy(self, state, epsilon=0):
        '''
        Select action according to function
        '''
        weighted_value = state[:-2] / self.alpha
        action = np.argmax(weighted_value)

        return action

    def pick_action(self, state, step):
        action = self.greedy(state)
        return action
