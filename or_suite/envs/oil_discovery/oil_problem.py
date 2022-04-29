"""An oil environment also over [0,1].  

Here the agent interacts with the environment by picking a location to travel to, 
paying a cost of travel, and receiving a reward at the new location."""

import numpy as np
import gym
from gym import spaces
import math

from .. import env_configs


class OilEnvironment(gym.Env):
    """
        An oil discovery problem on the metric space [0,1]^k for some power k.  

        Here the state space and the action space
        are given to have the same dimension.

        Methods:
            get_config() : Returns the config dictionary used to initialize the environment.
            render(mode) : (UNIMPLEMENTED) Renders the environment in the mode passed in; 'human' is the only mode currently supported.
            close() : (UNIMPLEMENTED) Closes the window where the rendering is being drawn.

        Attributes:
            epLen: The (int) number of time steps to run the experiment for.
            oil_prob (lambda function): A function taken as input a state, action and timestep, and outputting a reward for moving agent to that location
            cost_param (float): The parameter regulating the cost for moving the agent from one location to another
            noise_variance (lambda function): A function taken as input state, action, and timestamp, and outputting the noise added on to moving the agent
            starting_state: An int list containing the starting locations for the agent.
            action_space: (Gym.spaces Box) Actions must be the location to move the agent.
            observation_space: (Gym.spaces Box) The location of the agent.
        """

    metadata = {'render.modes': ['human']}

    def __init__(self, config=env_configs.oil_environment_default_config):

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
        """Reset the environment to its original settings."""
        self.timestep = 0
        self.state = self.starting_state
        return self.state

    def get_config(self):
        return self.config

    def step(self, action):
        """
        Move one step in the environment.

        Args:
            action: The chosen action; int.
        Returns:
            double, int, 0/1:
            reward: double; the reward.

            newState: int; the new state.

            done: 0/1; the flag for end of the episode.
        """
        if isinstance(action, np.ndarray):
            action = action.astype(np.float32)
        assert self.action_space.contains(action)
        # print('state: ' + str(self.state))
        # print('action: ' + str(action))
        reward = min(1.0, max(self.oil_prob(self.state, action, self.timestep) -
                     self.cost_param*np.sum(np.abs(self.state - action)), 0))
        # print('reward: ' + str(reward))

        newState = np.minimum(1, np.maximum(0, action + np.random.normal([0 for _ in range(
            self.dim)], np.sqrt(self.noise_variance(self.state, action, self.timestep)))), dtype=np.float32)

        # newState = min(1, max(0, action + np.random.normal(0, np.sqrt(self.noise_variance(self.state, action, self.timestep)))))
        # newState = action
        if self.timestep != self.epLen - 1:
            done = False
        else:
            done = True

        self.state = newState
        self.timestep += 1

        return self.state, reward,  done, {}
