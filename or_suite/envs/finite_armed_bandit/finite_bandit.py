
import numpy as np
import gym
from gym import spaces
import math

from .. import env_configs




class FiniteBanditEnvironment(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env for a finite armed bandit problem
  """

  metadata = {'render.modes': ['human']}


  def __init__(self, config=env_configs.finite_bandit_default_config):
        '''
        For a more detailed description of each parameter, see the readme file
        
        epLen - number of time steps
        arm_means - means for each of the arms
        '''
        super(FiniteBanditEnvironment, self).__init__()

        self.config = config
        self.epLen = config['epLen']
        self.arm_means = config['arm_means']

        # specifies action space
        self.action_space = spaces.Discrete(len(self.arm_means))

        # The definition of the observation space is the same as the action space
        self.observation_space = spaces.MultiDiscrete(np.ones((len(self.arm_means)))*self.epLen)
        self.starting_state = np.zeros((len(self.arm_means)))
        self.state = self.starting_state


  def reset(self):
        """
        Reinitializes variables and returns the starting state
        """
        # Initialize the timestep
        self.timestep = 0
        self.state = self.starting_state

        return self.starting_state

  def get_config(self):
      return self.config

  def step(self, action):
        '''
        Move one step in the environment

        Args:
        action - arm to pull
        Returns:
            reward - float - reward based on the action chosen
            newState - list - new state of the system
            done - 0/1 - flag for end of the episode
        '''
        old_state = self.state

        # Update the state of the system according to the action taken and change 
        old_state[action] += 1
        newState = old_state

        reward = np.random.binomial(1, self.arm_means[action])

        info = {'reward' : reward}

        if self.timestep != self.epLen - 1:
            done = False
        else:
            done = True

        self.state = newState
        self.timestep += 1

        return self.state, reward,  done, info


  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()

  def close(self):
    pass

