import numpy as np
import cvxpy as cp
from .. import Agent


''' Agent which implements several heuristic algorithms'''
class hopeguardrailAgent(Agent):

    def __init__(self, epLen, env_config, scale):
        '''args:
            epLen - number of steps
            func - function used to decide action
            env_config - parameters used in initialization of environment
            data - all data observed so far
        '''

        self.env_config = env_config
        self.num_types = env_config['weight_matrix'].shape[0]
        self.num_resources = self.env_config['weight_matrix'].shape[1]
        self.budget_remaining = np.copy(self.env_config['init_budget'])
        self.scale = scale

        #print('Starting Budget: ' + str(self.current_budget))
        
        self.epLen = epLen
        self.data = []
        self.first_allocation_done = False
        self.exp_endowments, self.var_endowments = self.get_expected_endowments()
        self.prob, self.solver = self.generate_cvxpy_solver()
        self.lower_sol = np.zeros((self.num_types,self.num_resources))
        self.upper_sol = np.zeros((self.num_types, self.num_resources))
        print('Mean and variance endomwnets:')
        print(self.exp_endowments, self.var_endowments)

        #print("R")
        #print(self.rel_exp_endowments)

    def generate_cvxpy_solver(self):
        """
        Creates a generic solver to solve the offline resource allocation problem
        
        Inputs: 
            num_types - number of types
            num_resources - number of resources
        Returns:
            prob - CVXPY problem object
            solver - function that solves the problem given data
        """
        num_types = self.num_types
        num_resources = self.num_resources
        x = cp.Variable(shape=(num_types,num_resources))
        sizes = cp.Parameter(num_types, nonneg=True)
        weights = cp.Parameter((num_types, num_resources), nonneg=True)
        budget = cp.Parameter(num_resources, nonneg=True)
        objective = cp.Maximize(cp.log(cp.sum(cp.multiply(x, weights), axis=1)) @ sizes)
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
    
    def get_lower_upper_sol(self,init_sizes):
        """
        uses solver to get the lower and upper
        """
        budget = self.env_config['init_budget']
        weights = self.env_config['weight_matrix']
        n = self.env_config['num_rounds']
        tot_size =  np.sum(self.exp_endowments[:,1:], axis=1)
        future_size = init_sizes + tot_size


        conf_bnd = np.sqrt(np.max(self.var_endowments, axis=1)*np.mean(self.exp_endowments, axis=1)*(n-1))


        # print(future_size)
        
        lower_exp_size = future_size*(1 + np.max(np.sqrt(conf_bnd) / future_size))
        _, lower_sol = self.solver(lower_exp_size, weights, budget)


        c = (1 / (n**(self.scale)))*(1 +  np.max(np.sqrt(conf_bnd) / future_size)) -  np.max(np.sqrt(conf_bnd) / future_size)
        # print('Scaling Constant')
        # print(c)
        upper_exp_size = future_size*(1 - c)
        _, upper_sol = self.solver(upper_exp_size, weights, budget)

        # print(lower_exp_size)
        # print(upper_exp_size)
        # print(budget)
        return lower_sol, upper_sol


    def get_expected_endowments(self,N=1000):
        """
        Monte Carlo Method for estimating Expectation of type distribution using N realizations
        Only need to run this once to get expectations for all locations

        Returns: 
        rel_exp_endowments: matrix containing expected proportion of endowments for location t
        """
        num_types = self.env_config['weight_matrix'].shape[0]
        exp_size = np.zeros((num_types, self.env_config['num_rounds']))
        var_size = np.zeros((num_types, self.env_config['num_rounds']))

        # print(num_types)
        # print(self.env_config['num_rounds'])
        for t in range(self.env_config['num_rounds']):
            cur_list = []
            for _ in range(N):
                obs_size = self.env_config['type_dist'](t)
                exp_size[:, t] += obs_size
                cur_list.append(obs_size)
            exp_size[:, t] = (1/N)*exp_size[:, t]
            var_size[:, t] = np.var(np.asarray(cur_list), axis=0)
        # print(exp_size)
        return exp_size, var_size

    def reset(self):
        # resets data matrix to be empty
        self.current_budget = np.copy(self.env_config['init_budget'])

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
        self.current_budget = np.copy(self.env_config['init_budget'])


    def pick_action(self, state, step):

        budget_remaining = state[:self.num_resources]
        sizes = state[self.num_resources:]
        num_remaining = self.env_config['num_rounds'] - step

        if not self.first_allocation_done:
            self.lower_sol, self.upper_sol = self.get_lower_upper_sol(sizes)
            self.first_allocation_done = True
            print('Lower and Upper Solutions:')
            print(self.lower_sol, self.upper_sol)




        conf_bnd = np.sqrt(np.max(self.var_endowments, axis=1)*np.mean(self.exp_endowments, axis=1)*num_remaining)


        budget_required = budget_remaining - np.matmul(sizes, self.upper_sol) - np.matmul(np.sum(self.exp_endowments[:, (step+1):], axis=1) + conf_bnd, self.lower_sol) > 0
        budget_index = budget_remaining - np.matmul(sizes, self.lower_sol) > 0

        allocation = budget_required * self.upper_sol \
                + (1 - budget_required)*budget_index*self.lower_sol \
                + (1 - budget_required) * (1 - budget_index) * np.array([budget_remaining / np.sum(sizes)])
        
        return allocation
