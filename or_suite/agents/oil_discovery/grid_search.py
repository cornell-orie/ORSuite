import numpy as np
import sys
from .. import Agent


class grid_searchAgent(Agent):
    """
    TODO: Add description here

    Methods:
        reset() : clears data and call_locs which contain data on what has occurred so far in the environment
        update_config() : (UNIMPLEMENTED)
        pick_action(state, step) : TODO add documentation

    Attributes:
        epLen: (int) number of time steps to run the experiment for
    """

    def __init__(self, epLen, dim=1):
        """
        Args:
            epLen: (int) number of time steps to run the experiment for

        """

        # Saving parameters like the epLen, dimension of the space
        self.epLen = epLen 
        self.dim = dim

        # Current bounds for the upper and lower estimates on where the maximum value is
        self.upper = np.ones((epLen, dim))
        self.lower = np.zeros((epLen, dim))

        # Estimates obtained for the "perturbed" values
        self.perturb_estimates = np.zeros(epLen, 2*dim) # TODO: needs to depend on epLEn
        self.midpoint_value = np.zeros(epLen)
        self.dim_index = [0 for _ in range(self.epLen)] # TODO: Need to keep track of which dimension you are checking for the different steps

        # Indicator of "where" we are in the process, i.e. selecting the midpoint, doing small perturbations, etc
        self.eps = 1e-7
        self.select_midpoint = [True for _ in range(self.epLen)]


    def reset(self):
        # Resets data and call_locs arrays to be empty
        self.upper = np.ones((self.epLen, self.dim))
        self.lower = np.zeros((self.epLen, self.dim))

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Adds newObs, the most recently observed state, to data
            adds the most recent call arrival, found in info['arrival'] to call_locs.'''
        if self.select_midpoint[timestep]: # If we selected the midpoint
            self.midpoint_value[timestep] = reward # Store value of midpoint estimate
            self.select_midpoint[timestep] = False # Switch to sampling the purturbed values

        else:
            self.perturb_estimates[timestep, self.dim_index] = reward # stores the observed reward
            self.dim_index[timestep] += 1

            if self.dim_index[timestep] == 2*self.dim: # finished getting all the purturbed estimates
                # TODO: Update the midpoint?
                # Get max value from the dim_list
                # Figure out how to cut the upper and lower estimates, and continue?
                self.upper[timestep] = self.upper[timestep] / 2 # just doing something stupid here for now
                self.dim_index[timestep] = 0
                self.select_midpoint[timestep] = True

        return

    def update_policy(self, k):
        '''Update internal policy based upon records.

        Not used, because a greedy algorithm does not have a policy.'''

        # Greedy algorithm does not update policy
        pass


    def pick_action(self, state, step):
        if self.select_midpoint[step]:
            action = (self.upper[step] + self.lower[step]) / 2
        else:
            # One line calculation of purturbation I think?
            # Gets the dimension index, mods it by 2 to get a 0,1 value, takes (-1) to the power
            # so the sign switches from positive and negative
            # 
            p_location = np.zeros(self.dim)
            p_location[int(np.floor(self.dim_index / 2))] = 1
            perturbation = np.zeros(self.dim) + (-1)**(np.mod(self.dim_index, 2))*p_location
            action = (self.upper[step] + self.lower[step]) / 2
        return action
