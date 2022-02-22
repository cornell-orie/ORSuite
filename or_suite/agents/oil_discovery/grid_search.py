import numpy as np
import sys
from .. import Agent


class grid_searchAgent(Agent):
    """
    TODO: Add description here

    Methods:
        reset() : clears data and call_locs which contain data on what has occurred so far in the environment
        update_config() : (UNIMPLEMENTED)
        pick_action(state, step) : TODO

    Attributes:
        epLen: (int) number of time steps to run the experiment for
    """

    def __init__(self, epLen, dim=1):
        """
        Args:
            epLen: (int) number of time steps to run the experiment for

        """
        self.epLen = epLen
        self.dim = dim
        self.upper = np.ones((epLen, dim))
        self.lower = np.zeros((epLen, dim))
        self.select_midpoint = True


    def reset(self):
        # Resets data and call_locs arrays to be empty
        self.upper = np.ones((self.epLen, self.dim))
        self.lower = np.zeros((self.epLen, self.dim))

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Adds newObs, the most recently observed state, to data
            adds the most recent call arrival, found in info['arrival'] to call_locs.'''

        self.call_locs.append(info['arrival'])
        return

    def update_policy(self, k):
        '''Update internal policy based upon records.

        Not used, because a greedy algorithm does not have a policy.'''

        # Greedy algorithm does not update policy
        pass


    def pick_action(self, state, step):
        if self.select_midpoint:
            action = (self.upper[step] + self.lower[step]) / 2
        else:
            action = (self.upper[step] + self.lower[step]) / 2
        return action
