"""
A simple vaccine distribution environment. 
 
Agent interacts w/ environment by picking a vaccine priority group order for vaccine distribution to a population over a set time period. 
Vaccines are distributed to the first priority group until there are no vaccines left or no people in the first priority group. 
If there are still vaccines available, they are distributed to the next priority group. 
We go down the priority list until vaccine supply is exhausted or there are no vaccine candidates left. 
There is a penalty for new infections in this time period.
Only 4 groups are considered (e.g. medical workers, essential non-medical workers, low-risk, others):
    - TOTAL population size and vaccine supply for each time period is kept constant. 
    - Each group is split into susceptible (S), asymptomatically infected (A)
    - We keep track of agreggate mildly symptomatically infected (I) and hospitalized (H) individuals. 
"""

# Implementation of a basic RL environment for discrete spaces.

import numpy as np
import gym
from gym import spaces
#import math
#import random
from .. import dynamics_model_4groups as dm4g
from .. import env_configs


class VaccineEnvironment(gym.Env):
    """
    A simple vaccine distribution environment with a discrete action and observation space.

    Attributes:
        config : (dict) Contains the parameters needed to set up the vaccine environment.
        epLen : (int) Number of time steps in an episode.
        vaccines : (int) Current number of vaccines available per time step.
        priority: (list) Current priority order of the groups.
        parameters: (dict) Contains the parameters needed for the dynamics.
        total_pop: (int) Total population size.
        state: (np.array) Current state of the environment.
        starting_state: (np.array) Beginning state of the environment.
        timestep: (int) The step we are on.
        action_space: (spaces.Discrete)
        all_priority_orders: (list)
        observation_space: (spaces.MultiDiscrete)

    """
    # don't worry about this, has to do with how gym exports text/info to the termial
    metadata = {'render.modes': ['human']}

    def __init__(self, config=env_configs.vaccine_default_config1):
        """
        Creates a VaccineEnvironment object.

        Arguments: 
            config: dictionary with the following keys (and corresponding values)

                - epLen : (int) Number of time steps.

                - starting_state : (np.array) initial population group sizes; should contain 11 entries > [S1 S2 S3 S4 A1 A2 A3 A4 I H R]

                - parameters : (dict) of parameter values to pass to dynamics model with the folowing keys and values:

                        - contact_matrix: (np.array of floats) Contact rates between susceptible people in each class and the infected people.

                        - P: (np.array of floats) P = [p1 p2 p3 p4] where pi = Prob(symptomatic | infected) for a person in class i.

                        - H: (np.array of floats) H = [h1 h2 h3 h4] where hi = Prob(hospitalized | symptomatic) for a person in class i.

                        - beta: (float) Recovery rate.

                        - gamma: (int) Vaccination rate.

                        - vaccines: (int) Number of vaccine available for this time period.

                        - priority: (list of chars) Vaccination priority order of the four groups.

                        - time_step: (float) Number of units of time you want the simulation to run for e.g. if all your rates are per day and you want to simulate 7 days, time_step = 7.


        Action Space (finite):

            - Set of all possible actions (priority orders in this case).

            - An action is an index i and the corresponding permutation is all_priority_orders[i].

            - 0 corresponds to [], which means no priority order.

            - See vaccine_allocation_readme.ipynb for more detail.




        Observation space (finite):
        A tuple of integer values representing certain population stats. 
        See vaccine_allocation_readme.ipynb for more detail.
        """

        self.config = config
        self.epLen = config['epLen']
        self.vaccines = config['parameters']['vaccines']
        self.priority_order = config['parameters']['priority']
        self.parameters = config['parameters']
        self.total_pop = np.sum(config['starting_state'])
        self.state = config['starting_state']
        self.starting_state = config['starting_state']
        self.timestep = 0

        self.action_space = spaces.Discrete(25)
        self.all_priority_orders = [[], ["1", "2", "3", "4"], ["2", "1", "3", "4"], ["3", "1", "2", "4"], ["1", "3", "2", "4"],
                                    ["2", "3", "1", "4"], ["3", "2", "1", "4"], [
                                        "4", "2", "1", "3"], ["2", "4", "1", "3"],
                                    ["1", "4", "2", "3"], ["4", "1", "2", "3"], [
                                        "2", "1", "4", "3"], ["1", "2", "4", "3"],
                                    ["1", "3", "4", "2"], ["3", "1", "4", "2"], [
            "4", "1", "3", "2"], ["1", "4", "3", "2"],
            ["3", "4", "1", "2"], ["4", "3", "1", "2"], [
                "4", "3", "2", "1"], ["3", "4", "2", "1"],
            ["2", "4", "3", "1"], ["4", "2", "3", "1"], ["3", "2", "4", "1"], ["2", "3", "4", "1"]]

        # The obersvation/state space is a spaces.MultiDiscrete object
        self.observation_space = spaces.MultiDiscrete(([self.total_pop+1]*11))

    def reset(self):
        """
        Reinitializes variables and returns the starting state.
        """
        self.timestep = 0
        self.state = self.starting_state
        return self.starting_state

    def get_config(self):
        """
        Returns the current configuration.
        """
        return self.config

    def step(self, action):
        """
        Moves one step in the environment.

        Arguments:
            action: int; the chosen action.

        Returns:
            double, np.array of integers, 0/1, dict:
            reward: double; the reward based on chosen action.

            newState: np.array of integers; the new state.

            done: 0/1; the flag for end of the episode.

            info: dict; information we can use to plot things related to disease dynamics.
        """
        assert self.action_space.contains(action), "Action is invalid"

        old_state = self.state
        # print('old_state' , old_state)

        self.priority_order = self.all_priority_orders[action]
        self.parameters['priority'] = self.priority_order

        newState, info = dm4g.dynamics_model(self.parameters, self.state)
        # print('New state' , newState)

        # 'reward' is number of new infections times -1
        reward = float(-1*newState[len(newState)-1])

        if self.timestep != (self.epLen-1):
            done = False
        else:
            done = True

        self.state = newState
        self.timestep += 1

        return self.state, reward,  done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        pass
