"""
A price competition environment over
"""

import numpy as np
import gym
from gym import spaces
import math
from .. import env_configs
from gym.envs.classic_control import rendering
# import pyglet
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
renderdir = os.path.dirname(currentdir)
sys.path.append(renderdir)
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

# ------------------------------------------------------------------------------


class PriceCompetitionEnvironment(gym.Env):
    """
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        """

        Args: 
            config: A (dict) dictionary containing the parameters required to set up a metric ambulance environment.
            epLen: The (int) number of time steps to run the experiment for.
            value_covariants_distribution: The distribution of the float list demographic covariants for the customers. 
            starting_state: A float list containing the starting covariants of consumers.
        """
        
        self.config = config
        self.epLen = config['epLen']
        self.value_covariants_distribution = config['value_covariants_distribution']
        self.item_0_covariants = config['item_0_covariants']
        self.item_1_covariants = config['item_1_covariants']
        self.starting_state = config['starting_state']
        self.state = np.array(self.starting_state, dtype=np.float32)
        self.timestep = 0

        # The action space is a box with two seller's prices for two items.
        self.action_space = spaces.Box(low=0.0, high=1000.0, shape=(2,2),
                                    dtype=np.float32)

        # The observation space is a box with the current demographic covariants for the customer.
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(13,),
                                    dtype=np.float32)


    def reset(self):
        """Reinitializes variables and returns the starting state."""

        self.timestep = 0
        self.state = self.starting_state

        return self.starting_state

    def get_config(self):
        return self.config

    def value_covariants(self):
        num = len(self.value_covariants_distribution)
        ans = []
        for i in range(num):
            [mu, sigma] = self.value_covariants_distribution[i,:]
            value = np.random.normal(mu, sigma, 1)
            ans.append(value)
        return ans
    
    def calculate_value(self, value_covariants):
        item0 = np.inner(value_covariants,self.item_0_covariants)
        item1 = np.inner(value_covariants,self.item_1_covariants)
        return [item0, item1]

    def step(self, action):
        """
        Move one step in the environment.

        Args:
            action: A 2*2 box contains two seller's prices for two items.
        Returns:
            reward: A 2*2 box of the revenues for two sellers on two items representing the reward based on the action chosen.

            newState: A 13*1 box of the current demographic covariants for the customers representing the state of the environment.

            done: A bool flag indicating the end of the episode.
        """

        if isinstance(action, spaces.box):
            action = action.astype(np.float32)
        assert self.action_space.contains(action)

        [seller0_item0, seller1_item0] = action[0,:]

        [seller0_item1, seller1_item1] = action[1,:]

        value = self.calculate_value(self.state)

        reward = np.array([[0.0,0.0],[0.0,0.0]])

        if value[0] < min(seller0_item0, seller1_item0):
            reward = reward
        elif seller0_item0 < seller1_item0:
            reward += np.array([[seller0_item0,0.0],[0.0,0.0]])
        else:
            reward += np.array([[0.0,seller1_item0],[0.0,0.0]])

        if value[1] < min(seller0_item1, seller1_item1):
            reward = reward
        elif seller0_item1 < seller1_item1:
            reward += np.array([[seller0_item1,0.0],[0.0,0.0]])
        else:
            reward += np.array([[0.0,seller1_item1],[0.0,0.0]])

        value_covariants = self.value_covariants()
        new_state = value_covariants

        if self.timestep != self.epLen - 1:
            done = False
        else:
            done = True

        self.state = new_state
        self.timestep += 1

        assert self.observation_space.contains(self.state)

        return self.state, reward, done
