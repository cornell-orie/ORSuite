import numpy as np
import sys
from .. import Agent


class base_surgeAgent(Agent):

    def __init__(self, r, S):
        self.r = r
        self.S = S

        # S is the goal inventory level

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.config = config
        lead_times = config['lead_times']
        self.offset = config['max_inventory']
        self.max_order = config['max_order']

        # Doesn't include longest lead time (assuming lead times sorted in non-decreasing order)

        # TODO: Figure out problem with action not being part of observation space
        # Set up tests for 1 and 2 suppliers
        # Run Stable Baselines (uncomment SB line)
        # Look into linear programming solvers ( CVXPY, PuLP, or others)

    def pick_action(self, obs, h):
        '''Select an action based upon the observation'''
        # Step 1, extract I_t from obs
        inventory = obs[-1] - self.offset

        order_amount = min(self.max_order, max(0, self.S - inventory))
        # print(
        #     f'Current order_amount: {order_amount} and inventory {inventory}')

        # order_amount = min(self.max_order, min(
        #     self.offset - 1, max(0, self.S - inventory)))
        action = np.asarray(self.r+[order_amount])
        # action = [self.r, order_amount]
        return action

    def update_parameters(self, param):
        if len(param[0]) == 0:
            self.r = []
        else:
            self.r = [param[0]]
        self.S = param[1]
        print(self.r, self.S)
