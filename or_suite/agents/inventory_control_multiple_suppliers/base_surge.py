import numpy as np
import sys
from .. import Agent


class base_surgeAgent(Agent):

    def __init__(self, r, S):
        self.r = r
        self.S = S

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.config = config
        # TODO: Need to figure out which  have shorter lead times, used for the self.r action
        # Make r a vector of shorter lead times
        # Have longest lead time used for order_amount = max(...)
        # Assume config is sorted by increasing lead times; can add check to env for this

    def pick_action(self, obs, h):
        '''Select an action based upon the observation'''
        # Step 1, extract I_t from obs

        inventory = obs[-1]
        order_amount = max(0, self.S - inventory)

        action = [self.r, order_amount]
        return action
