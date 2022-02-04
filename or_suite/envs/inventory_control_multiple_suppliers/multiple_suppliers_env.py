import gym
import numpy as np
import sys

from .. import env_configs


class DualSourcingEnvironment(gym.Env):
    """
    An environment with a variable number of suppliers, each with their own lead time and cost.

    Attributes:
        lead_times: The array of ints representing the lead times of each supplier.
        supplier_costs: The array of ints representing the costs of each supplier.
        hold_cost: The int holding cost.
        backorder_cost: The int backorder cost.
        epLen:  The int number of time steps to run the experiment for.
        max_order: The maximum value (int) that can be ordered from each supplier.
        max_inventory: The maximum value (int) that can be held in inventory.
        timestep: The (int) timestep the current episode is on.
        starting_state: An int list containing enough indices for the sum of all the lead times, plus an additional index for the initial on-hand inventory.
        action_space: (Gym.spaces MultiDiscrete) Actions must be the length of the number of suppliers. Each entry is an int corresponding to the order size. 
        observation_space: (Gym.spaces MultiDiscrete) The environment state must be the length of the of the sum of all lead times plus one. Each entry corresponds to the order that will soon be placed to a supplier. The last index is the current on-hand inventory.
        neg_inventory: A bool that says whether the on-hand inventory can be negative or not.
    """

    def __init__(self, config):
        """
        Args:
            config: A dictionary containt the following parameters required to set up the environment:
                lead_times: array of ints representing the lead times of each supplier
                supplier_costs: array of ints representing the costs of each supplier
                demand_dist: The random number sampled from the given distribution to be used to calculate the demand
                hold_cost: The int holding cost.
                backorder_cost: The int backorder cost.
                epLen: The episode length
                max_order: The maximum value (int) that can be ordered from each supplier
                max_inventory: The maximum value (int) that can be held in inventory
                starting_state: An int list containing enough indices for the sum of all the lead times, plus an additional index for the initial on-hand inventory.
                neg_inventory: A bool that says whether the on-hand inventory can be negative or not. 
            """
        self.lead_times = config['lead_times']
        self.supplier_costs = config['supplier_costs']
        self.config = config
        self.demand_dist = config['demand_dist']
        self.hold_cost = config['hold_cost']
        self.backorder_cost = config['backorder_cost']
        L_total = sum(self.lead_times)
        self.starting_state = config['starting_state']
        if self.starting_state == None:
            self.starting_state = [0] * (L_total + 1)
        self.max_order = config['max_order']
        self.max_inventory = config['max_inventory']
        self.starting_state[-1] = self.max_inventory

        self.neg_inventory = config['neg_inventory']

        if self.neg_inventory:  # inventory can be negative
            self.starting_state[-1] = self.max_inventory
        else:
            self.starting_state[-1] = 0

        self.state = np.asarray(self.starting_state)

        self.action_space = gym.spaces.MultiDiscrete(
            [self.max_order+1]*len(self.lead_times))


        if self.neg_inventory:  # inventory can be negative
            self.observation_space = gym.spaces.MultiDiscrete(
                [self.max_order+1]*(L_total)+[2 * self.max_inventory + 1])
        else:  # inventory is only positive
            self.observation_space = gym.spaces.MultiDiscrete(
                [self.max_order+1]*(L_total)+[self.max_inventory + 1])

        # Check to see if cost and lead time vectors match
        assert len(self.supplier_costs) == len(self.lead_times)
        self.timestep = 0
        self.epLen = config['epLen']

        metadata = {'render.modes': ['human']}

    def get_config(self):
        return self.config

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
        
        assert self.action_space.contains(
            action), "Action, {},  not part of action space".format(action)
        demand = self.demand_dist(self.timestep)
        newState = self.new_state_helper(self.state, action)
        newState[-1] = newState[-1] - demand
        if self.neg_inventory:  # Inventory can be negative
            newState[-1] = max(- self.max_inventory, min(newState[-1] -
                               self.max_inventory, self.max_inventory)) + self.max_inventory
            amount_neg = 0
        else:  # Inventory is only positive
            if newState[-1] < 0:
                amount_neg = newState[-1]
            else:
                amount_neg = 0
            newState[-1] = max(0,
                               min(newState[-1], self.max_inventory))
        self.state = newState.copy()

        assert self.observation_space.contains(self.state)

        reward = self.reward(self.state) + self.backorder_cost * amount_neg

        self.timestep += 1
        done = self.timestep == self.epLen

        return self.state, float(reward), done, {'demand': demand}

    def get_config(self):
        return self.config

    # Auxilary function computing the reward

    def reward(self, state):
        """
        Reward is calculated in three components:
            - First component corresponds to the cost for ordering amounts from each supplier
            - Second component corresponds to paying a holding cost for extra inventory after demand arrives
            - Third component corresponds to a back order cost for unmet demand
        """
        total = 0
        sum_previous_lead_times = 0
        for i in range(0, len(self.lead_times)):
            total += self.supplier_costs[i] * \
                state[self.lead_times[i] - 1 + sum_previous_lead_times]
            sum_previous_lead_times += self.lead_times[i]

        if self.neg_inventory:  # Inventory can be negative
            return -(total + self.hold_cost*max(state[-1] - self.max_inventory, 0) + self.backorder_cost*max(-(state[-1] - self.max_inventory), 0))
        else:  # Inventory is only positive
            return -(total + self.hold_cost*max(state[-1], 0) + self.backorder_cost*max(-(state[-1]), 0))

    # Auxilary function
    def new_state_helper(self, state, action):
        running_L_sum = 1
        vec = []
        inventory_add_sum = state[-1]
        for i in range(0, len(self.lead_times)):
            inventory_add_sum += state[running_L_sum - 1]
            vec = np.hstack(
                (vec, state[running_L_sum: running_L_sum - 1 + self.lead_times[i]], action[i]))
            running_L_sum += self.lead_times[i]
        return np.hstack((vec, inventory_add_sum)).astype(int)

    def render(self, mode='human'):
        outfile = sys.stdout if mode == 'human' else super(
            DualSourcingEnvironment, self).render(mode=mode)
        outfile.write(np.array2string(self.state)+'\n')

    def reset(self):
        """Reinitializes variables and returns the starting state."""
        self.state = np.asarray(self.starting_state)
        self.timestep = 0
        return self.state
