'''
All agents should inherit from the Agent class.
'''
import numpy as np
import sys
from .. import Agent
import cvxpy as cp


airline_default_config = {
    'epLen': epLen,
    'f': np.asarray([1., 2.]),
    'A': np.transpose(np.asarray([[2., 3., 2.], [3., 0., 1.]])),
    'starting_state': np.asarray([10., 10., 10.]),
    'P': np.asarray([[1/3, 1/3] for _ in range(epLen+1)])
}




class bayes_selectorAgent(Agent):

    def __init__(self, epLen):
        self.epLen = epLen
        pass

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.config = config
        return
        


    def pick_action(self, obs):
        '''Select an action based upon the observation'''
        # use the config to populate vector of the demands
        print(f'Triggering a new LP solve')
        num_type = len(self.config['f'])
        expect_type = np.sum(self.config['P'], axis=1)
            # gets the expected number of customer arrivals
        
        x = cp.Variable(num_type)
        objective = cp.Maximize(self.config['f'].T@x)
        constraints = []
        constraints += [0 <= x]
        constraints += [x <= num_type]

        constraints += [self.config['A'] @ x <= obs]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        print("\nThe optimal value is", prob.value)
        print("A solution x is")
        print(x.value)

        # enforcing rounding rule here, add a trigger to do the other version somehow as well
        action = np.asarray([1 if x.value[i] / num_type[i] >= 1/2 else 0 for i in range(num_type)])
        return action
