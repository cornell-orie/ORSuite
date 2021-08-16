"""Sequential Resource Allocation Problem for n locations with K commodities with Donations. 
Currently reward is Nash Social Welfare but in the future will integrate more options 
to determine a fair allocation """

import numpy as np
import gym
from gym import spaces
import math
from .. import env_configs
from .resource_allocation import ResourceAllocationEnvironment


class ResourceAllocationDonationsEnvironment(gym.Env):
    """Custom Environment that follows gym interface."""
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['human']}
    # Define constants for clearer code

    def __init__(self, config=env_configs.resource_allocation_default_config):
        """
        Initializes the Sequential Resource Allocation Environment.

        Args:
            weight_matrix: Weights predefining the commodity needs for each type, every row is a type vector.
            K: Number of commodities.
            num_rounds: Number of locations (also the length of an episode).
            init_budget: Amount of each commodity the principal begins with.
            type_dist: Function determining the number of people of each type at a location.
            u: Utility function, given an allocation x and a type theta, u(x,theta) is how good the fit is.

            donation_size_dist: Distribution of donation sizes whenever there is one.
            donation_schedule: List detailing the timesteps in which donations are received.
        """
        super(ResourceAllocationEnvironment, self).__init__()

        self.config = config

        self.weight_matrix = config['weight_matrix']
        self.num_types = config['weight_matrix'].shape[0]
        self.num_commodities = config['K']
        self.epLen = config['num_rounds']
        self.budget = config['init_budget']
        self.type_dist = config['type_dist']
        self.utility_function = config['utility_function']

        self.donation_size_dist = config['donation_size_dist']
        self.donation_schedule = config['set_donation_schedule'](self.epLen)

        self.starting_state = np.concatenate(
            [config['init_budget'], self.type_dist(0)])
        # print(np.concatenate([config['init_budget'],self.type_dist(0)]))

        self.state = self.starting_state
        self.timestep = 0

        # Action space will be choosing Kxn-dimensional allocation matrix (represented as a vector)
        self.action_space = spaces.Box(low=0, high=max(self.budget),
                                       shape=(self.num_commodities*self.num_types,), dtype=np.float32)
        # First K entries of observation space is the remaining budget, next is the number of each type at the location
        self.observation_space = spaces.Box(low=0, high=np.inf,
                                            shape=(self.num_commodities+self.num_types,), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        Returns: np.array
        """
        # Initialize the timestep
        self.timestep = 0
        self.state = self.starting_state
        return self.starting_state

    def get_config(self):
        """Returns: the environment config (dict)."""
        return self.config

    def step(self, action):
        """
        Move one step in the environment

        Args:
            action: A matrix; the chosen action (each row how much to allocate to prev location).
        Returns:
            double, int, 0/1, dict:
            reward: double; the reward.

            newState: int; the new state.

            done: 0/1; the flag for end of the episode.

            info: dict; any additional information. 
        """

        # subdividing state of (b,N) into the two components
        old_budget = self.state[:self.num_commodities]
        old_type = self.state[self.num_commodities:]

        # reshaping the allocation into a matrix
        allocation = np.reshape(
            np.array(action), (self.num_types, self.num_commodities))

        # determines if the allocation is valid, i.e. algorithm is able to allocate the allocation
        # to each of the types, based on the number of people of each type

        # print('Allocation: ' + str(allocation))
        # print('Budget: ' + str(old_budget))
        # print('Types: ' + str(old_type))

        # print('New Budget: ' + str(old_budget-np.matmul(old_type, allocation)))

        if np.min(old_budget - np.matmul(old_type, allocation)) >= -.0005:

            reward = (1/np.sum(old_type))*sum(
                [old_type[theta]*np.log(self.utility_function(allocation[theta, :],
                                        self.weight_matrix[theta, :])) for theta in range(self.num_types)]
            )

            # updates the budget by the old budget and the allocation given
            new_budget = old_budget-np.matmul(old_type, allocation)

            if self.timestep != self.epLen - 1:
                done = False
            else:
                done = True

        else:  # algorithm is allocating more than the budget, output a negative infinity reward
            print('Out of Budget!')
            reward = -np.inf
            done = True
            new_budget = old_budget

        new_type = self.type_dist(self.timestep)

        info = {'type': new_type}

        donation = np.zeros(self.num_commodities)
        if self.timestep in self.donation_schedule:
            donation = self.donation_size_dist(self.timestep)

        self.state = np.concatenate([new_budget+donation, new_type])

        self.action_space = spaces.Box(low=0, high=max(new_budget),
                                       shape=(self.num_commodities*self.num_types,), dtype=np.float32)

        self.timestep += 1

        return self.state, reward,  done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        pass
