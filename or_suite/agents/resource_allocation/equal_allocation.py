import numpy as np
from .. import Agent


class equalAllocationAgent(Agent):
    """ 
    Equal Allocation Agent subdivides the initial budget equally among all locations. 
    Each location-specific allocation will be further subdivided (so as to create the 
    matrix of allocation) by relative proportion of the types present at location i.

    Methods:
        get_expected_endowments(N=1000) : MCM for estimating Expectation of type distribution using N realizations.
        reset() : Resets bounds of agent to reflect upper and lower bounds of metric space.
        update_config(env, config) : Updates environment configuration dictionary.
        update_obs(obs, action, reward, newObs, timestep, info) : Add observation to records.
        update_policy(k) : Update internal policy based upon records.
        pick_action(state, step) : Move agent to midpoint or perturb current dimension

    Attributes:
        num_types (int) : Number of types
        num_resources (int) : Number of commodities
        current_budget (int) : Amount of each commodity the principal begins with.
        epLen (int) : Number of locations (also the length of an episode).
        data (list) : All data observed so far
        rel_exp_endowments (matrix) : Matrix containing expected proportion of endowments for location t
    """

    def __init__(self, epLen, env_config):
        """
        Initialize equal_allocation agent

        Args:
            epLen: number of steps
            env_config: parameters used in initialization of environment
        """
        self.env_config = env_config
        self.num_types = env_config['weight_matrix'].shape[0]
        self.num_resources = self.env_config['weight_matrix'].shape[1]
        self.current_budget = np.copy(self.env_config['init_budget']())
        self.epLen = epLen
        self.data = []
        self.rel_exp_endowments = self.get_expected_endowments()

    def get_expected_endowments(self, N=1000):
        """
        Monte Carlo Method for estimating Expectation of type distribution using N realizations
        Only need to run this once to get expectations for all locations
        Returns: rel_exp_endowments: matrix containing expected proportion of endowments for location t
        """
        num_types = self.env_config['weight_matrix'].shape[0]
        exp_size = np.zeros((num_types, self.env_config['num_rounds']))
        for t in range(self.env_config['num_rounds']):
            for _ in range(N):
                obs_size = self.env_config['type_dist'](t)
                exp_size[:, t] += obs_size
            exp_size[:, t] = (1/N)*exp_size[:, t]

        return exp_size

    def reset(self):
        # resets data matrix to be empty
        self.current_budget = np.copy(self.env_config['init_budget']())
        self.data = []

    def update_config(self, env, config):
        '''Updates environment configuration dictionary'''
        self.env_config = config
        return

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''
        self.data.append(newObs)
        return

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        self.current_budget = np.copy(self.env_config['init_budget']())

    def pick_action(self, state, step):
        '''
        Returns allocation of resources based on budget times expectation of type 
        distribution at current step divided by summation of expectation of type
        distribution over all future steps

        Args: 
            state : vector with first K entries denoting remaining budget, 
                    and remaining n entires denoting the number of people of each type that appear
            step : timestep

        Returns: matrix where each row is a K-dimensional vector denoting how 
            much of each commodity is given to each type
        '''
        num_types = self.env_config['weight_matrix'].shape[0]
        sizes = state[self.num_resources:]
        action = np.zeros((num_types, self.num_resources))

        for typ in range(num_types):
            action[typ, :] = (self.current_budget/sizes[typ]) * (self.rel_exp_endowments[typ,
                                                                                         step] / np.sum(self.rel_exp_endowments[typ, step:]))

        self.current_budget -= np.sum([action[typ, :] * sizes[typ]
                                      for typ in range(num_types)])

        # prevent negative budget due to rounding
        self.current_budget = list(
            map(lambda x: max(x, .0005), self.current_budget))

        # prevent distribution beyond max limit of state (offset in calculation due to rounding)
        max_state = np.amax(state)
        for r in range(len(action)):
            for c in range(len(action[r, :])):
                if action[r, c] > max_state:
                    action[r, c] = max_state

        return action
