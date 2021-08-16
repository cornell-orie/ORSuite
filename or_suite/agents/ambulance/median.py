import numpy as np
import sys
from .. import Agent


class medianAgent(Agent):
    """
    Agent that implements a median-like heuristic algorithm for the metric environment.
    The data on previous call locations is sorted by location and split into k quantiles, where k is the 
    number of ambulances. The midpoint of each quantile is chosen as the location for one
    of the ambulances.

    Methods:
        reset() : clears data and call_locs which contain data on what has occurred so far in the environment
        update_config() : (UNIMPLEMENTED)
        pick_action(state, step) : locations are chosen by finding the the midpoints of 
            each of k quantiles of the arrival data sorted by location, where k is the 
            number of ambulances

    Attributes:
        epLen: (int) number of time steps to run the experiment for
        data: (float list list) a list of all the states of the environment observed so far
        call_locs: (float list) the locations of all calls observed so far

    """

    def __init__(self, epLen):
        """
        Args:
            epLen: (int) number of time steps to run the experiment for

        """
        self.epLen = epLen
        self.data = []
        self.call_locs = []

    def reset(self):
        # Resets data and call_locs arrays to be empty
        self.data = []
        self.call_locs = []

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Adds newObs, the most recently observed state, to data
            adds the most recent call arrival, found in info['arrival'] to call_locs.'''

        # Adds the most recent state obesrved in the environment to data
        self.data.append(newObs)

        # Adds the most recent arrival location observed to call_locs
        self.call_locs.append(info['arrival'])
        return

    def update_policy(self, k):
        '''Update internal policy based upon records.

        Not used, because a greedy algorithm does not have a policy.'''

        # Greedy algorithm does not update policy
        self.greedy = self.greedy

    def greedy(self, state, timestep, epsilon=0):
        """
        This algorithm sorts all previous arrivals, and splits them into k quantiles,
        where k is the number of ambulances. It then finds the median of each quantile,
        and positions an ambulance there.
        """

        if len(self.data) == 0:
            return state
        else:
            num_ambulance = len(self.data[0])
            # print(num_ambulance)
            left_points = [(1 / num_ambulance)*i for i in range(num_ambulance)]
            quantiles = []
            for j in range(len(left_points)):
                if j == len(left_points) - 1:
                    quantiles.append(
                        ((1 - left_points[j]) / 2) + left_points[j])
                else:
                    quantiles.append(
                        ((left_points[j+1] - left_points[j]) / 2) + left_points[j])
            # print(quantiles)
            action = np.quantile(np.asarray(self.call_locs), quantiles)
            # print(action)
            return action

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
