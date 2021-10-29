'''
All agents should inherit from the Agent class.
'''
import numpy as np
import sys
from .. import Agent
import cvxpy as cp


class bayes_selectorAgent(Agent):

    def __init__(self, epLen, round_flag=True):
        self.epLen = epLen
        self.round_flag = round_flag
        pass

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.config = config
        return
        
    def reset(self):
        pass


    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Adds newObs, the most recently observed state, to data
            adds the most recent call arrival, found in info['arrival'] to call_locs.'''
        return


    def pick_action(self, obs, timestep):
        '''Select an action based upon the observation'''
        # use the config to populate vector of the demands
        num_type = len(self.config['f'])
        expect_type = np.sum(self.config['P'][timestep:,:],axis=0)
            # gets the expected number of customer arrivals
        x = cp.Variable(num_type)
        objective = cp.Maximize(self.config['f'].T @ x)
        constraints = []
        constraints += [0 <= x]
        constraints += [x <= expect_type]

        constraints += [self.config['A'] @ x <= obs]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        # enforcing rounding rule here, add a trigger to do the other version somehow as well
        if self.round_flag:
            action = np.asarray([1 if x.value[i] / expect_type[i] >= 1/2 else 0 for i in range(num_type)])
        else:
            action = np.asarray([np.random.binomial(1, np.minimum(1, np.maximum(0, x.value[i] / expect_type[i])), size=None) for i in range(num_type)])
        return action