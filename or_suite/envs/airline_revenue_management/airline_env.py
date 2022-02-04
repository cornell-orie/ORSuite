import gym
import numpy as np
import sys
import copy
import math

from .. import env_configs


class AirlineRevenueEnvironment(gym.Env):
    """
    An environment representing the airline revenue management problem

    Attributes:
        A: The 2-D float array representing the resource consumption.
        f: The float array representing the revenue per class.
        P: The float array representing the distribution over arrivals.
        epLen: The int number of time steps to run the experiment for.
        starting_state: The float array representing the number of available seats on each flight.
        timestep: The int timestep the current episode is on.
        action_space: (Gym.spaces MultiDiscrete) Actions must be binary arrays of the length of the number of customers.
        observation_space: (Gym.spaces MultiDiscrete) States must be float arrays of the length of the number of flights.

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        """
        Args:
            A: The 2-D float array representing the resource consumption.
            f: The float array representing the revenue per class.
            P: The float array representing the distribution over arrivals.
            epLen: The int number of time steps to run the experiment for.
            starting_state: The float array representing the number of available seats on each flight.
            """
        # Initializes model parameters based on a configuration dictionary
        self.config = config
        self.A = config['A']  # resource consumption
        self.f = config['f']  # revenue per class
        self.P = config['P']  # distribution over arrivals
        self.epLen = config['epLen']  # length of episode
        self.starting_state = config['starting_state']  # starting state

        self.config = config

        # Defines state and action spaces, sets current state to be starting_state
        self.action_space = gym.spaces.MultiBinary(self.A.shape[1])
        sstate = np.asarray(self.starting_state)+1
        self.observation_space = gym.spaces.MultiDiscrete(sstate)
        self.state = np.asarray(self.starting_state)
        self.timestep = 0

    def get_config(self):
        return self.config

    # Resets environment to initial state
    def reset(self):
        """Reinitializes variables and returns the starting state."""
        self.state = np.asarray(self.starting_state)
        self.timestep = 0
        return self.state

    # Defines one step of the MDP, returning the new state, reward, whether time horizon is finished, and a dictionary of information

    def step(self, action):
        """Move one step in the environment."""

        assert self.action_space.contains(action)

        # Sample customer arrival
        pDist = np.append(
            np.copy(self.P[self.timestep, :]), 1 - np.sum(self.P[self.timestep, :]))
        customer = np.random.choice(range(self.A.shape[1]+1), 1, p=pDist)[0]

        # Check if valid action
        valid = True
        for j in range(len(action)):
            nState = np.copy(self.state) - self.A[:, j]*action[j]
            if not len(nState[nState < 0]) == 0:
                valid = False

        # Given a valid action
        newState = np.copy(self.state)
        reward = 0.
        if (not customer == self.A.shape[1]) and valid:
            if action[customer] == 1:
                newState = np.copy(self.state) - self.A[:, customer]
                reward = float(self.f[customer])
        self.state = newState
        episode_over = False
        self.timestep += 1
        if self.timestep == self.epLen:
            episode_over = True
        return self.state, reward, episode_over, {'customer': customer}

    def get_config(self):
        return self.config
