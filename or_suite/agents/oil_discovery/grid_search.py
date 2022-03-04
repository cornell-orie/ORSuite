import numpy as np
import sys
from .. import Agent


class grid_searchAgent(Agent):
    """
    Agents that uses a bisection-method heuristic algorithm to find location of most oil.

    Methods:
        reset() : clears data and call_locs which contain data on what has occurred so far in the environment
        update_config() : (UNIMPLEMENTED)
        pick_action(state, step) : If upper and lower bounds set, move agent to midpoint. Else, perturb location.

    Attributes:
        epLen: (int) number of time steps to run the experiment for
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
        self.eps = 1e-7
        self.select_midpoint = [True for _ in range(self.epLen)]

    def reset(self):
        # Resets data and call_locs arrays to be empty
        self.upper = np.ones((self.epLen, self.dim))
        self.lower = np.zeros((self.epLen, self.dim))

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        ''' If no perturbations needed, update reward to be midpoint. Else, cut upper and lower
            bounds based on higher rewards from perturbation. '''

        # If we selected the midpoint in prev step
        if self.select_midpoint[timestep]:
            # Store value of midpoint estimate
            self.midpoint_value[timestep] = reward
            # Switch to sampling the purturbed values
            self.select_midpoint[timestep] = False

        else:
            # stores the observed reward
            # print()
            # print("upper: ", self.upper.flatten())
            # print("lower: ", self.lower.flatten())
            # print("timestep", timestep)
            # print("reward", reward)
            # print("dim_ind", self.dim_index)
            self.perturb_estimates[timestep,
                                   self.dim_index[timestep]] = reward
            self.dim_index[timestep] += 1

            # print("pert estimates", self.perturb_estimates)
            # print(self.dim_index)

            if self.dim_index[timestep] > 0 and self.dim_index[timestep] % 2 == 0:
                # print("pert estimates", self.perturb_estimates)
                # print("upperl: ", self.upper.flatten())
                # print("lowerl: ", self.lower.flatten())
                # print()
                # 2 perturbations (forward and back) in each dimension

                midpoint = (self.upper[timestep] +
                            self.lower[timestep]) / 2

                # compare perturbation forward (self.dim_index[timestep]-2) with
                # perturbation backwards (self.dim_index[timestep]-1) in each dimension
                if self.perturb_estimates[timestep, self.dim_index[timestep]-2] \
                        > self.perturb_estimates[timestep, self.dim_index[timestep]-1]:
                    # if lower perturbation has higher reward, move lower up
                    self.lower[timestep][(
                        self.dim_index[timestep]-2)] = midpoint
                else:
                    self.upper[timestep][(
                        self.dim_index[timestep]-2)] = midpoint

                self.dim_index[timestep] = 0
                self.select_midpoint[timestep] = True

        return

    def update_policy(self, k):
        '''Update internal policy based upon records.

        Not used, because a greedy algorithm does not have a policy.'''

        # Greedy algorithm does not update policy
        pass

    def pick_action(self, state, step):
        ''' If upper and lower bounds are updated based on perturbed values, move agent to midpoint.
            Else, perturb area surrounding current midpoint. '''

        # action taken at step h is used to maximize the step h+1 oil function
        next_step = step+1 if step+1 < self.epLen else step

        # print()
        if self.select_midpoint[step]:
            # should be plus 1, add extra if statement to prevent index out of bounds
            action = (self.upper[next_step] + self.lower[next_step]) / 2
        else:
            # One line calculation of perturbation I think?
            # Gets the dimension index, mods it by 2 to get a 0,1 value, takes (-1) to the power
            # so the sign switches from positive and negative
            p_location = np.zeros(self.dim)
            p_location[int(np.floor(self.dim_index[step] / 2))] = 1
            perturbation = np.zeros(
                self.dim) + (-1)**(np.mod(self.dim_index[step], 2))*p_location
            # print("p_loc", p_location)
            # print("dim", self.dim)
            # print("eq", (-1)**(np.mod(self.dim_index[step], 2))*p_location)
            # print("upper", self.upper[step])
            # print("lower", self.lower[step])
            # print("pert", perturbation)
            # print("pert calc", perturbation *
            #   (self.upper[step] - self.lower[step])/2)

            # limit perturbation to distance from midpoint to upper or lower
            action = (self.upper[next_step] + self.lower[next_step]) / 2 + \
                (perturbation*(self.upper[next_step] -
                 self.lower[next_step])/2)

        # print("act", action)
        return action
