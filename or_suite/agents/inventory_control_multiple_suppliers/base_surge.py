import numpy as np
import sys
from .. import Agent


class base_surgeAgent(Agent):

    def __init__(self, r, S):
        # TODO: Add documentation in the correct form, similar to the other algorithms outlining the definition of r and S and their interpretation
        self.r = r
        self.S = S

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.config = config
        if config['neg_inventory']:
            self.offset = config['max_inventory']
        else:
            self.offset = 0
        self.max_order = config['max_order']

    def pick_action(self, obs, h):
        '''Select an action based upon the observation'''
        # Step 1, extract I_t from obs
        inventory = obs[-1] - self.offset

        order_amount = min(self.max_order, max(0, self.S - inventory))
        # TODO: Max(0, asdf) important? Is it ever realized?
        # adjust for the stuff which is currently coming in from previous lead time step?

        action = np.asarray(self.r+[order_amount])
        return action

    def update_parameters(self, param):
        self.r = param[0]
        self.S = param[1]
        #print(self.r, self.S)
