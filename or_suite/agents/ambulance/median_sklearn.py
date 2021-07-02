import numpy as np
import sys
import sklearn_extra.cluster

import sys
from .. import Agent


class median_sklearnAgent(Agent):
    """
    Agent that implements a k-medoid heuristic algorithm for the metric ambulance environment

    Methods:
        reset() : Clears data and call_locs which contain data on what has occurred so far in the environment
        update_config() : (UNIMPLEMENTED)
        pick_action(state, step) : Locations are chosen by finding the k-medoids in the 
            accumulated arrival data, where k is the number of ambulances, using 
            sci-kit learn's k-medoids algorithm

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
        For the first iteration, choose the starting state
        After that, choose locations for the ambulances that are most centrally
        located to the locations of previous calls using the k-medoids algorithm
        For more details about the k-medoids algorithm, see the readme document
        for the ambulance environment or the sci-kit learn documentation
        """
        num_ambulance = len(self.data[0])
        action = []
        if len(self.call_locs) > num_ambulance:
            reshaped_call_locs = np.asarray(self.call_locs).reshape(-1, 1)
            clusters = sklearn_extra.cluster.KMedoids(
                n_clusters=num_ambulance, max_iter=50).fit(reshaped_call_locs)
            action = np.asarray(clusters.cluster_centers_).reshape(-1,)
        else:
            action = np.full(num_ambulance, np.median(self.call_locs))

        return action

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
