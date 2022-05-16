import numpy as np
import cvxpy as cp
from .. import Agent


class hopeguardrailAgent(Agent):
    """ 
    Hope Guardrail provides upper and lower thresholds on budget distribution
    calculated by solving the primal-dual paradigm of Eisenberg-Gale Convex Progam

    Methods:
        generate_cvxpy_solver() : Creates a generic solver to solve the offline resource allocation problem.
        get_lower_upper_sol(init_size) : Uses solver to get the lower and upper "guardrails" on budget distribution
        get_expected_endowments(N=1000) : MCM for estimating Expectation of type distribution using N realizations.
        reset() : resets bounds of agent to reflect upper and lower bounds of metric space.
        update_config(env, config) : Updates environment configuration dictionary.
        update_obs(obs, action, reward, newObs, timestep, info) : Add observation to records.
        update_policy(k) : Update internal policy based upon records.
        pick_action(state, step) : move agent to midpoint or perturb current dimension

    Attributes:
        num_types (int) : Number of types
        num_resources (int) : Number of commodities
        budget_remaining (int) : Amount of each commodity the principal begins with.
        scale (int) : Hyperparameter to be used in calculating threshold 
        epLen (int) : Number of locations (also the length of an episode).
        data (list) : All data observed so far
        first_allocation_done (bool) : Flag that if false, gets upper and lower thresh
        conf_const (int) : Hyperparameter for confidence bound
        exp_endowments (list) : Matrix containing expected proportion of endowments for location t
        stdev_endowments (list) : Matrix describing variance of exp_endowments
        prob (cvxpy object) : CVXPY problem object
        solver (lambda function) : Function that solves the problem given data
        lower_sol (np.array) : Matrix of lower threshold 
        upper_sol (np.array) : Matrix of upper threshold
    """

    def __init__(self, epLen, env_config, scale):
        '''
        Initialize hope_guardrail agent

        Args:
            epLen: number of steps
            env_config: parameters used in initialization of environment
            scale: hyperparameter to be used in calculating threshold 
        '''
        self.env_config = env_config
        self.num_types = env_config['weight_matrix'].shape[0]
        self.num_resources = self.env_config['weight_matrix'].shape[1]
        self.budget_remaining = np.copy(self.env_config['init_budget']())
        self.scale = scale
        self.epLen = epLen
        self.data = []
        self.first_allocation_done = False
        self.conf_const = 2
        self.from_data = env_config['from_data']
        self.exp_endowments, self.stdev_endowments = self.get_expected_endowments()
        self.prob, self.solver = self.generate_cvxpy_solver()
        self.lower_sol = np.zeros((self.num_types, self.num_resources))
        self.upper_sol = np.zeros((self.num_types, self.num_resources))

    def generate_cvxpy_solver(self):
        """
        Creates a generic solver to solve the offline resource allocation problem

        Returns:
            prob - CVXPY problem object
            solver - function that solves the problem given data
        """
        num_types = self.num_types
        num_resources = self.num_resources
        x = cp.Variable(shape=(num_types, num_resources))
        sizes = cp.Parameter(num_types, nonneg=True)
        weights = cp.Parameter((num_types, num_resources), nonneg=True)
        budget = cp.Parameter(num_resources, nonneg=True)
        objective = cp.Maximize(
            cp.log(cp.sum(cp.multiply(x, weights), axis=1)) @ sizes)
        constraints = []
        constraints += [0 <= x]
        for i in range(num_resources):
            constraints += [x[:, i] @ sizes <= budget[i]]
        # constraints += [x @ sizes <= budget]
        prob = cp.Problem(objective, constraints)

        def solver(true_sizes, true_weights, true_budget):
            sizes.value = true_sizes
            weights.value = true_weights
            budget.value = true_budget
            prob.solve()
            return prob.value, np.around(x.value, 5)
        return prob, solver

    def get_lower_upper_sol(self, init_sizes):
        """
        Uses solver to get the lower and upper "guardrails" on budget distribution

        Args:
            init_sizes (list) : vector containing the number of each type at each location
        """
        budget = self.env_config['init_budget']()
        weights = self.env_config['weight_matrix']
        n = self.env_config['num_rounds']

        tot_size = np.sum(self.exp_endowments[:, 1:], axis=1)
        future_size = init_sizes + tot_size

        conf_bnd = self.conf_const * np.sqrt(np.max(self.stdev_endowments, axis=1)
                                             * np.mean(self.exp_endowments, axis=1)*(n-1))

        lower_exp_size = future_size * \
            (1 + np.max(conf_bnd / future_size))
        _, lower_sol = self.solver(lower_exp_size, weights, budget)

        c = (1 / (n**(self.scale)))*(1 + np.max(conf_bnd /
                                                future_size)) - np.max(conf_bnd / future_size)

        upper_exp_size = future_size*(1 - c)

        _, upper_sol = self.solver(upper_exp_size, weights, budget)

        return lower_sol, upper_sol

    def get_expected_endowments(self, N=1000):
        """
        Monte Carlo Method for estimating Expectation of type distribution using N realizations
        Only need to run this once to get expectations for all locations

        Returns: 
            rel_exp_endowments - matrix containing expected proportion of endowments for location t
        """
        num_types = self.env_config['weight_matrix'].shape[0]
        exp_size = np.zeros((num_types, self.env_config['num_rounds']))
        var_size = np.zeros((num_types, self.env_config['num_rounds']))

        for t in range(self.env_config['num_rounds']):
            cur_list = []
            for _ in range(N):
                obs_size = self.env_config['type_dist'](t)
                exp_size[:, t] += obs_size
                cur_list.append(obs_size)
            exp_size[:, t] = (1/N)*exp_size[:, t]
            var_size[:, t] = np.var(np.asarray(cur_list), axis=0)

        return exp_size, np.sqrt(var_size)

    def reset(self):
        ''' Resets data matrix to be empty '''
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
        Returns allocation of resources based on calculated upper and lower solutions 

        Args: 
            state : vector with first K entries denoting remaining budget, 
                    and remaining n entires denoting the number of people of each type that appear
            step : timestep

        Returns: matrix where each row is a K-dimensional vector denoting how 
                much of each commodity is given to each type
        '''
        if step == 0:
            self.current_budget = np.copy(self.env_config['init_budget']())
            if self.from_data:
                mean, stdev = self.env_config['type_dist'](-2)
                self.exp_endowments = np.transpose(mean)
                self.stdev_endowments = np.transpose(stdev)
            sizes = state[self.num_resources:]
            self.lower_sol, self.upper_sol = self.get_lower_upper_sol(
                sizes)

        budget_remaining = state[:self.num_resources]
        sizes = state[self.num_resources:]
        num_remaining = self.env_config['num_rounds'] - step

        conf_bnd = np.sqrt(np.max(self.stdev_endowments, axis=1)
                           * np.mean(self.exp_endowments, axis=1)*num_remaining)

        budget_required = budget_remaining - np.matmul(sizes, self.upper_sol) - np.matmul(
            np.sum(self.exp_endowments[:, (step+1):], axis=1) + conf_bnd, self.lower_sol) > 0

        budget_index = budget_remaining - np.matmul(sizes, self.lower_sol) > 0

        allocation = budget_required * self.upper_sol \
            + (1 - budget_required) * budget_index * self.lower_sol \
            + (1 - budget_required) * (1 - budget_index) * \
            np.array([budget_remaining / np.sum(sizes)])

        allocation = np.array([list(map(lambda x: max(x, 0.0), values))
                               for values in allocation])

        return allocation
