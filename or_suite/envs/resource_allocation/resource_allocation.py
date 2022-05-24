"""
Sequential Resource Allocation Problem for n locations with K commodities.

A ResourceAllocationEnvironment where agent iterates through locations and 
receives a reward of Nash Social Welfare based on the resources it allocates,
conditioned that allocation is within budget
"""

import numpy as np
import gym
from gym import spaces
from .. import env_configs


class ResourceAllocationEnvironment(gym.Env):
    """Custom Environment that follows gym interface.

    This is a simple resource allocation environment modeling a fair online allocation 

     Methods:
        get_config() : Returns the config dictionary used to initialize the environment.
        reset() : Resets environment to original starting state and timestep to 0
        step(action) : Takes in allocation as action subtracts from budget, calculates reward, and updates action space
        render(mode) : (UNIMPLEMENTED) Renders the environment in the mode passed in; 'human' is the only mode currently supported.
        close() : (UNIMPLEMENTED) Closes the window where the rendering is being drawn.

    Attributes:
        weight_matrix (list) : Weights predefining the commodity needs for each type, every row is a type vector.
        num_types (int) : Number of types
        num_commodities (int) : Number of commodities
        epLen (int) : Number of locations (also the length of an episode).
        budget (int) : Amount of each commodity the principal begins with.
        type_dist (lambda function) : Function determining the number of people of each type at a location.
        utility_function (lambda function) : Utility function, given an allocation x and a type theta, u(x,theta) is how good the fit is.
        starting_state (np.array) : Tuple (represented as list concat) of initial budget and type distribution.
        timestep (int) : Step that is executed in an episode of an iteration.
        action_space : (Gym.spaces Box) Action space represents the K x n allocation matrix.
        observation_space : (Gym.spaces Box) The first K entries to the observation space is remaining budget, 
                            with the remaining spaces filled by the number of each type at each location.

    """

    metadata = {'render.modes': ['human']}

    def __init__(self, config=env_configs.resource_allocation_default_config):
        """Inits RideshareGraphEnvironment with the given configuration.

        Args:
            config: A dictionary containing the initial configuration of the resource allocation environment.
        """

        super(ResourceAllocationEnvironment, self).__init__()

        self.config = config
        self.weight_matrix = config['weight_matrix']
        self.num_types = config['weight_matrix'].shape[0]
        self.num_commodities = config['K']
        self.epLen = config['num_rounds']
        self.type_dist = config['type_dist']
        self.utility_function = config['utility_function']
        self.budget = config['init_budget']()
        self.from_data = config['from_data']
        self.starting_state = np.concatenate(
            [self.budget, self.type_dist(0)]).astype(np.float32)
        self.MAX_VAL = config['MAX_VAL']

        self.state = self.starting_state
        self.timestep = 0

        self.action_space = spaces.Box(low=0, high=max(self.budget),
                                       shape=(self.num_types, self.num_commodities), dtype=np.float32)
        # First K entries of observation space is the remaining budget, next is the number of each type at the location
        self.observation_space = spaces.Box(low=0, high=np.append(self.budget, [self.MAX_VAL]*self.num_types),
                                            shape=(self.num_commodities+self.num_types,), dtype=np.float32)

    def reset(self):
        """
        Requires: the observation must be a numpy array
        Returns: np.array
        """
        # if FBST data, then reset index in env_config
        if self.from_data:
            self.type_dist(-1)

        self.action_space = spaces.Box(low=0, high=max(self.budget),
                                       shape=(self.num_types, self.num_commodities), dtype=np.float32)

        self.state = np.concatenate(
            [self.budget, self.type_dist(0)]).astype(np.float32)
        self.budget = self.config['init_budget']()
        self.timestep = 0

        return self.state

    def get_config(self):
        """Returns: the environment config (dict)."""
        return self.config

    def step(self, action):
        """
        Move one step in the environment.

        Args:
            action: A matrix; the chosen action (each row how much to allocate to prev location).

        Returns:
            double, int, 0/1, dict:
            reward (double) : the reward.
            newState (int): the new state.
            done (bool) : the flag for end of the episode.
            info (dict) : any additional information.
        """
        if isinstance(action, np.ndarray):
            action = action.astype(np.float32)
        assert self.action_space.contains(action)
        # subdividing state of (b,N) into the two components
        old_budget = self.state[:self.num_commodities]
        old_type = self.state[self.num_commodities:]

        # reshaping the allocation into a matrix
        allocation = np.reshape(
            np.array(action), (self.num_types, self.num_commodities))

        # determines if the allocation is valid, i.e. algorithm is able to allocate the allocation
        # to each of the types, based on the number of people of each type
        if np.min(old_budget - np.matmul(old_type, allocation)) >= -.0005:

            reward = max(-100, (1/np.sum(old_type))*sum(
                [old_type[theta]*np.log(self.utility_function(allocation[theta, :],
                                        self.weight_matrix[theta, :])) for theta in range(self.num_types)]))

            # updates the budget by the old budget and the allocation given
            if self.timestep != self.epLen - 1:
                # temp budget in case of rounding errors
                new_budget = old_budget-np.matmul(old_type, allocation)
                done = False

            else:
                new_budget = self.budget
                done = True
        else:  # algorithm is allocating more than the budget, output a negative infinity reward
            print('Out of Budget!')
            reward = -100
            done = False
            new_budget = old_budget

        new_type = self.type_dist(self.timestep)

        info = {'type': new_type}

        self.state = np.concatenate([new_budget, new_type]).astype(np.float32)
        self.action_space = spaces.Box(low=0, high=max(new_budget),
                                       shape=(self.num_types, self.num_commodities), dtype=np.float32)
        self.timestep += 1

        return self.state, float(reward),  done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        pass
