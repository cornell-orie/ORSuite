import numpy as np

import sklearn_extra.cluster

import sys
from .. import Agent


class modeAgent(Agent):
    """
    Agent that implements a mode heuristic algorithm for the ambulance graph environment

    Methods:
        reset() : clears data and call_locs which contain data on what has occurred so far in the environment
        update_config() : (UNIMPLEMENTED)
        pick_action(state, step) : locations are chosen by finding the k modes of the arrival data so far,
            where k is the number of ambulances

    Attributes:
        epLen: (int) number of time steps to run the experiment for
        data: (int list list) a list of all the states of the environment observed so far
        call_locs: (int list) the node locations of all calls observed so far

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
        '''
        Chooses the k nodes where calls have arrived most frequently in the past,
        where k is the number of ambulances.
        '''

        # For the first iteration, choose the starting state
        # After that, choose the locations where calls have occurred most frequently
        # in the past
        if len(self.data) == 0:
            return state
        else:
            num_ambulance = len(self.data[0])
            counts = np.bincount(self.call_locs)
            action = []
            for i in range(num_ambulance):
                mode = np.argmax(counts)
                action.append(mode)
                counts[mode] = 0
            return action

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
