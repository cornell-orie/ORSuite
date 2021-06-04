import numpy as np
from .. import Agent


''' Agent which implements several heuristic algorithms'''
class equalAllocationAgent(Agent):

    def __init__(self, epLen, env_config):
        '''args:
            epLen - number of steps
            func - function used to decide action
            env_config - parameters used in initialization of environment
            data - all data observed so far
        '''
        self.env_config = env_config

        self.num_types = env_config['weight_matrix'].shape[0]
        self.num_resources = self.env_config['weight_matrix'].shape[1]
        
        self.current_budget = np.copy(self.env_config['init_budget'])
        #print('Starting Budget: ' + str(self.current_budget))
        self.epLen = epLen
        self.data = []
        self.rel_exp_endowments = self.get_expected_endowments()
        #print("R")
        #print(self.rel_exp_endowments)



    def get_expected_endowments(self,N=1000):
        """
        Monte Carlo Method for estimating Expectation of type distribution using N realizations
        Only need to run this once to get expectations for all locations

        Returns: 
        rel_exp_endowments: matrix containing expected proportion of endowments for location t
        """
        num_types = self.env_config['weight_matrix'].shape[0]
        exp_size = np.zeros((num_types, self.env_config['num_rounds']))
        #print(num_types)
        #print(self.env_config['num_rounds'])
        for t in range(self.env_config['num_rounds']):
            for _ in range(N):
                obs_size = self.env_config['type_dist'](t)
                exp_size[:, t] += obs_size
            exp_size[:, t] = (1/N)*exp_size[:, t]

        return exp_size

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
        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''

        num_types = self.env_config['weight_matrix'].shape[0]
        sizes = state[self.num_resources:]
        action = np.zeros((num_types, self.num_resources))

        for typ in range(num_types):
            action[typ,:] = (self.env_config['init_budget'] / sizes[typ])*(self.rel_exp_endowments[typ, timestep] / np.sum(self.rel_exp_endowments))

        self.current_budget -= np.sum([action[typ,:] * sizes[typ] for typ in range(num_types)])
        #print('Allocation: ' + str(action))

        return action


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
