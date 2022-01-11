import numpy as np
import sys
from .. import Agent


class commandLineAgent(Agent):
    """
    Allows the user to act as the agent by entering locations for each of the ambulances through the command line. Only works with the metric environment.

    Methods:
        reset() : clears data and call_locs which contain data on what has occurred so far in the environment
        update_config() : (UNIMPLEMENTED)
        pick_action(state, step) : the user is prompted to enter a new location for each ambulance, and these locations are used as the action

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
        '''adds newObs, the most recently observed state, to data
            adds the most recent call arrival, found in info['arrival'] to call_locs'''

        # Adds the most recent state obesrved in the environment to data
        self.data.append(newObs)

        # Adds the most recent arrival location observed to call_locs
        self.call_locs.append(info['arrival'])
        return

    def update_policy(self, k):
        '''Update internal policy based upon records.

        Not used, because action is chosen by user.'''

        # Greedy algorithm does not update policy
        self.greedy = self.greedy

    def greedy(self, state, timestep, epsilon=0):
        """
        The user is prompted to choose an action by choosing a new location for each ambulance.
        """
        num_ambulance = len(state)
        action = np.zeros(num_ambulance)
        for ambulance in range(num_ambulance):
            done = False
            while(not done):
                text = "Where do you want to position ambulance " + \
                    str(ambulance+1) + "? (choose a number between 0 and 1)\n"
                new_loc = input(text)
                try:
                    new_loc = float(new_loc)
                    if new_loc < 0 or new_loc > 1:
                        raise ValueError
                    action[ambulance] = new_loc
                    done = True
                except ValueError:
                    print("Please enter a number between 0 and 1")

        return np.float32(action)

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
