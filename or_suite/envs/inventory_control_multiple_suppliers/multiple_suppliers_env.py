import gym
import numpy as np
import sys
from scipy.stats import poisson

from .. import env_configs


class DualSourcingEnvironment(gym.Env):
    """
    An environment with a variable number of suppliers, each with their own lead time and cost.

    Attributes:
        L: The array of ints representing the lead times of each supplier.
        c: The array of ints representing the costs of each supplier.
        lambda: The Poission distribution parameter.
        h: The int holding cost.
        b: The int backorder cost.
        epLen:  The int number of time steps to run the experiment for.
        max_order: The maximum value (int) that can be ordered from each supplier.
        max_inventory: The maximum value (int) that can be held in inventory.
        timestep: The (int) timestep the current episode is on.
        starting_state: An int list containing enough indices for the sum of all the lead times, plus an additional index for the initial on-hand inventory.
        action_space: (Gym.spaces MultiDiscrete) Actions must be the length of the number of suppliers. Each entry is an int corresponding to the order size. 
        observation_space: (Gym.spaces MultiDiscrete) The environment state must be the length of the of the sum of all lead times plus one. Each entry corresponds to the order that will soon be placed to a supplier. The last index is the current on-hand inventory.
    """
    def __init__(self, config):
        """
        Args:
            config: A dictionary containt the following parameters required to set up the environment:
                L: array of ints representing the lead times of each supplier
                c: array of ints representing the costs of each supplier
                lambda: distribution parameter
                h: holding cost
                b: backorder cost
                epLen: The episode length
                max_order: the maximum value (int) that can be ordered from each supplier
                max_inventory: the maximum value (int) that can be held in inventory
            """
        self.L = config['L']
        self.c = config['c']

        # self.Lr = config['Lr']
        # self.Le = config['Le']
        # self.cr = config['cr']
        # self.ce = config['ce']
        self.Lambda = config['lambda']
        self.h = config['h']
        self.b = config['b']
        #self.starting_state = [0] * (self.Lr + self.Le + 1)
        L_total = sum(self.L)
        self.starting_state = [0] * (L_total + 1)
        self.max_order = config['max_order']
        self.max_inventory = config['max_inventory']

        self.state = np.asarray(self.starting_state)
        self.action_space = gym.spaces.MultiDiscrete([self.max_order+1]*len(self.L))
        # self.observation_space = gym.spaces.MultiDiscrete(
        #     [self.max_order+1]*(self.Lr+self.Le)+[self.max_inventory])
        self.observation_space = gym.spaces.MultiDiscrete(
            [self.max_order+1]*(L_total)+[self.max_inventory])
        # Check to see if cost and lead time vectors match
        assert len(self.c) == len(self.L)
        self.timestep = 0
        self.epLen = config['epLen']

        self.nA = (self.max_order+1) ** 2

        metadata = {'render.modes': ['human']}

    def seed(self, seed=None):
        """Sets the numpy seed to the given value
        
        Args:
            seed: The int represeting the numpy seed."""
        np.random.seed(seed)
        self.action_space.np_random.seed(seed)
        return seed

    def step(self, action):
        """
        Move one step in the environment.
        
        Args:
            action: An int list of the amount to order from each supplier.
            
        Returns:
            float, int, bool, info:
            reward: A float representing the reward based on the action chosen.
            
            newState: An int list representing the new state of the environment after the action.
            
            done: A bool flag indicating the end of the episode.
            
            info: A dictionary containing extra information about the step. This dictionary contains the int value of the demand during the previous step"""
        assert self.action_space.contains(action)

        reward = self.r(self.state)

        demand = np.random.poisson(self.Lambda)
        newState = self.g(self.state, action)
        newState[-1] = newState[-1] - demand
        newState[-1] = max(0,
                           min(newState[-1], self.max_inventory))
        self.state = newState.copy()

        assert self.observation_space.contains(self.state)

        self.timestep += 1
        done = self.timestep == self.epLen

        return self.state, float(reward), done, {'demand': demand}

    # Auxilary function computing the reward
    def r(self, state):
        total = 0
        for i in range(0, len(self.L)):
            total += self.c[i]*state[self.L[i] - 1]
        return -(total + self.h*max(state[-1], 0) + self.b*max(-state[-1], 0))
        # Old function for two suppliers
        # return -(self.cr*state[self.Lr-1] + self.ce*state[self.Lr+self.Le-1] +
        #          self.h*max(state[-1], 0) + self.b*max(-state[-1], 0))

    # Auxilary function
    def g(self, state, action):
        running_L_sum = 1
        vec = []
        inventory_add_sum = state[-1]
        for i in range(0, len(self.L)):
            inventory_add_sum += state[running_L_sum - 1]
            vec = np.hstack(
                (vec, state[running_L_sum: running_L_sum - 1 + self.L[i]], action[i]))
            running_L_sum += self.L[i]
        return np.hstack((vec, inventory_add_sum)).astype(int)

        # Old function for two suppliers
        # return np.hstack([state[1:self.Lr], action[0], state[self.Lr+1:self.Lr+self.Le], action[1],
        #                  state[self.Lr+self.Le]+state[0]+state[self.Lr]]).astype(int)

    def render(self, mode='human'):
        outfile = sys.stdout if mode == 'human' else super(
            DualSourcingEnvironment, self).render(mode=mode)
        outfile.write(np.array2string(self.state)+'\n')

    def reset(self):
        """Reinitializes variables and returns the starting state."""
        self.state = np.asarray(self.starting_state)
        self.timestep = 0
        return self.state
