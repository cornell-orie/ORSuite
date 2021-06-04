'''An oil environment also over [0,1].  Here the agent interacts with the environment
by picking a location to travel to, paying a cost of travel, and receiving a reward at the new location.'''

import numpy as np
import gym
from gym import spaces
import math

from .. import env_configs


class OilEnvironment(gym.Env):

    metadata = {'render.modes': ['human']}


    def __init__(self, config = env_configs.oil_environment_default_config):
        """
        An oil discovery problem on the metric space [0,1]^k for some power k.  Here the state space and the action space
        are given to have the same dimension.
        
        Methods:
            reset() : resets the environment to its original settings
            get_config() : returns the config dictionary used to initialize the environment
            step(action) : takes an action from the agent and returns the state of the system after the next arrival
            render(mode) : (UNIMPLEMENTED) renders the environment in the mode passed in; 'human' is the only mode currently supported
            close() : (UNIMPLEMENTED) closes the window where the rendering is being drawn

        Attributes:
            epLen: (int) number of time steps to run the experiment for
            arrival_dist: (lambda) arrival distribution for calls over the observation space; takes an integer (step) and returns an integer that corresponds to a node in the observation space
            timestep: (int) the timestep the current episode is on
            starting_state: (int list) a list containing the starting locations for each ambulance
            action_space: (Gym.spaces Box) actions must be the location
            observation_space: (Gym.spaces Box) the location
    
        """
        self.config = config
        self.epLen = config['epLen']
        self.dim = config['dim']
        self.state = config['starting_state']
        self.starting_state = config['starting_state']
        self.timestep = 0
        self.oil_prob = config['oil_prob']
        self.cost_param = config['cost_param']
        self.noise_variance = config['noise_variance']

        self.observation_space = spaces.Box(low=0, high=1,
                                        shape=(self.dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1,
                                        shape=(self.dim,), dtype=np.float32)



    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = self.starting_state
        return self.state

    def get_config(self):
        return self.config


    def step(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action
        Returns:
            reward - double - reward
            newState - int - new state
            done - 0/1 - flag for end of the episode
        '''

        assert self.action_space.contains(action)
        # print('state: ' + str(self.state))
        # print('action: ' + str(action))
        reward = min(1, max(self.oil_prob(self.state, action, self.timestep) - self.cost_param*np.sum(np.abs(self.state - action)),0))
        # print('reward: ' + str(reward))

        newState = np.minimum(1, np.maximum(0, action + np.random.normal([0 for _ in range(self.dim)], np.sqrt(self.noise_variance(self.state, action, self.timestep)))))
       
        # newState = min(1, max(0, action + np.random.normal(0, np.sqrt(self.noise_variance(self.state, action, self.timestep)))))
        # newState = action
        if self.timestep != self.epLen - 1:
            done = False
        else:
            done = True
        

        self.state = newState
        self.timestep += 1

        return self.state, reward,  done, {}