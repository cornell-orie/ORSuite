import numpy as np
import random
import sys
from .. import Agent


class grid_searchAgent(Agent):
    """
    Agent that uses a bisection-method heuristic algorithm to the find location with 
    the highest probability of discovering oil. 

    Methods:
        reset() : resets bounds of agent to reflect upper and lower bounds of metric space
        update_config() : (UNIMPLEMENTED)
        update_obs(obs, action, reward, newObs, timestep, info) : record reward of current midpoint 
            or move bounds in direction of higher reward
        pick_action(state, step) : move agent to midpoint or perturb current dimension

    Attributes:
        epLen: (int) number of time steps to run the experiment for
        dim: (int) dimension of metric space for agent and environment
        upper: (float list list) matrix containing upper bounds of agent at each step in dimension
        lower: (float list list) matrix contianing lower bounds of agent at each step in dimension
        perturb_estimates: (float list list) matrix containing estimated rewards from perturbation in each dimension
        midpoint_value: (float list) list containing midpoint of agent at each step
        dim_index: (int list) list looping through various dimensions during perturbation
        select_midpoint: (bool list) list recording whether to take midpoint or perturb at given step
    """

    def __init__(self, epLen, dim=1):
        """
        Args:
            epLen: (int) number of time steps to run the experiment for
            dim: (int) dimension of metric space for agent and environment
        """

        # Saving parameters like the epLen, dimension of the space
        self.epLen = epLen
        self.dim = dim

        # Current bounds for the upper and lower estimates on where the maximum value is
        self.upper = np.ones((epLen, dim))
        self.lower = np.zeros((epLen, dim))

        # Estimates obtained for the "perturbed" values
        self.perturb_estimates = np.zeros((epLen, 2*dim))
        self.midpoint_value = np.zeros(epLen)
        self.dim_index = [0 for _ in range(self.epLen)]

        # Indicator of "where" we are in the process, i.e. selecting the midpoint, doing small perturbations, etc
        self.select_midpoint = [True for _ in range(self.epLen)]

    def reset(self):
        # Resets upper to array of ones, lower to array of zeros
        self.upper = np.ones((self.epLen, self.dim))
        self.lower = np.zeros((self.epLen, self.dim))

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        """
        If no perturbations needed, update reward to be value at midpoint. 
        Else, adjust upper or lower bound in the direction of higher 
        reward as determined by the perturbation step. Agent loops across
        each dimension separately, and updates estimated midpoint after each
        loop.
        """
        # If we selected the midpoint in prev step
        if self.select_midpoint[timestep]:
            # Store value of midpoint estimate
            self.midpoint_value[timestep] = reward
            # Switch to sampling the purturbed values
            self.select_midpoint[timestep] = False
        else:
            self.perturb_estimates[timestep, self.dim_index[timestep]] = reward
            self.dim_index[timestep] += 1

            if self.dim_index[timestep] > 0 and self.dim_index[timestep] % 2 == 0:
                # corresponding index of upper/lower bound matrix given self.dim_indx[timestep]
                bound_index = int(self.dim_index[timestep]/2 - 1)
                midpoint = (self.upper[timestep, bound_index] +
                            self.lower[timestep, bound_index]) / 2

                # compare pert forward with pert backwards in dimension of timestep
                pert_f = self.dim_index[timestep]-2
                pert_b = self.dim_index[timestep]-1
                if self.perturb_estimates[timestep, pert_f] > self.perturb_estimates[timestep, pert_b]:
                    # if lower perturbation has higher reward, move lower bound up
                    self.lower[timestep, bound_index] = midpoint
                else:
                    self.upper[timestep, bound_index] = midpoint

                # reset dim_index once perturbations completed in every dimension
                if self.dim_index[timestep] == 2*self.dim:
                    self.dim_index[timestep] = 0
                self.select_midpoint[timestep] = True

        return

    def update_policy(self, k):
        '''Update internal policy based upon records.

        Not used, because a greedy algorithm does not have a policy.'''

        # Greedy algorithm does not update policy
        pass

    def pick_action(self, state, step):
        """ 
        If upper and lower bounds are updated based on perturbed values, move agent to midpoint.
        Else, perturb dimension by factor equal to half the distance from each bound to midpoint. 
        """

        # action taken at step h is used to maximize the step h+1 oil function
        if step+1 < self.epLen:
            next_step = step+1
        # if last step, move agent to random location
        else:
            return np.random.rand(self.dim)

        if self.select_midpoint[step]:
            action = (self.upper[next_step] + self.lower[next_step]) / 2
        else:
            # Gets the dimension index, mods it by 2 to get a 0,1 value, takes (-1) to the power
            # so the sign switches from positive and negative
            p_location = np.zeros(self.dim)
            p_location[int(np.floor(self.dim_index[step] / 2))] = 1
            perturbation = np.zeros(
                self.dim) + (-1)**(np.mod(self.dim_index[step], 2))*p_location
            # perturb distance of 1/4 * width of dimension
            action = (self.upper[next_step] + self.lower[next_step]) / 2 + \
                (perturbation*(self.upper[next_step] -
                 self.lower[next_step])/(4))

        return action
