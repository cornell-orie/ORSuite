import numpy as np
import sys
from .. import Agent


class base_surgeAgent(Agent):
    """
    Uses a value, r, which is a vector of order amounts of length number of suppliers - 1, and an order-up-to-amount, S, which is used to calculate the order amount for the supplier with the greatest lead time.

    The base surge agent has 2 parameters, r and S. 
    Each action is expressed as [r,[orderamount]]. r is a vector of the order amounts for all suppliers except the one with the greatest lead time. 
    S represents the "order up to amount". 
    orderamount is calculated by calculating S - I where I is the current on-hand inventory.
    This value is then made 0 if it is negative or is reduced to the maxorder if it is greater. 
    This order amount is used for the supplier with the greatest lead time.

    Attributes:
        r: A vector of order amounts of length number of suppliers - 1.
        S: The order-up-to amount for the supplier with the greatest lead time.
        config: The dictionary of values used to set up the environment.
        offset: Either 0 or the value of the max_inventory. It is used to have correct order amounts when inventory is strictly positive or if it is positive and negative.
        max_order: The maximum order amount for every supplier.
  """

    def __init__(self, r, S):
        '''Initializes the agent with attributes r and S.

        Args:
            r: A vector of order amounts of length number of suppliers - 1.
            S: The order-up-to amount for the supplier with the greatest lead time.
        '''
        self.r = r
        self.S = S

    def update_config(self, env, config):
        ''' Update agent information based on the config__file

        Args:
            env: The environment being used.
            config: The dictionary of values used to set up the environment.'''
        self.config = config
        if config['neg_inventory']:
            self.offset = config['max_inventory']
        else:
            self.offset = 0
        self.max_order = config['max_order']

    def pick_action(self, obs, h):
        '''Select an action based upon the observation.

        Args:
            obs: The most recently observed state.
            h: Not used.

        Returns:
            list:
            action: The action the agent will take in the next timestep.'''
        # Step 1, extract I_t from obs
        inventory = obs[-1] - self.offset

        order_amount = min(self.max_order, max(0, self.S - inventory))
        # TODO: Max(0, asdf) important? Is it ever realized?
        # adjust for the stuff which is currently coming in from previous lead time step?

        action = np.asarray(self.r+[order_amount])
        return action

    def update_parameters(self, param):
        ''' Update the parameters, r and S.

        Args:
            param: A list of the form [r, S] where r is a list of integers and S is an integer.'''
        self.r = param[0]
        self.S = param[1]
        #print(self.r, self.S)
