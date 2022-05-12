"""Implementation of an RL environment in a discrete graph space.

A ridesharing environment over a simple graph. An agent interacts through the
environment by choosing a non-zero node to service a given rideshare request.
"""

import numpy as np
from numpy.random import default_rng
import gym
from gym import spaces
import networkx as nx
import math

from .. import env_configs


class RideshareGraphEnvironment(gym.Env):
    """Custom Rideshare Graph Environment that follows gym interface.

    This is a simple env where the requests are uniformly distributed across 
    nodes.

    Attributes:
        config: A dictionary containing the initial configuration of the 
            rideshare graph environment.
        epLen: An integer representing the total number of time steps.
        graph: An object containing nodes and edges; each edge has a travel time.
        num_nodes: An integer count of the number of nodes in the graph.
        starting_state: A vector representing the initial state of the 
            environment; the first K elements represent the number of cars
            at each node, and the final 2 elements represent the current 
            request that needs to be satisfied, i.e. node i to node j.
        state: A vector representing the state of the environment; the first K 
            elements represent the number of cars at each node, and the final 2 
            elements represent the current request that needs to be satisfied, 
            i.e. node i to node j.
        timestep: An integer representing the current timestep of the model.
        num_cars: An integer representing the number of cars in the model.
        lengths: A 2-dimensional symmetric array containing the distances 
            between each pair of nodes.
        request_dist: A vector consisting of the distribution used for selecting
            nodes when generating requests.
        reward: A lambda function to generate the reward.
        reward_fail: A lambda function to generate the reward when the RL
            agent fails; i.e. when a request is not satisfied.
        action_space: A discrete set of values the action can have; in this case
            the action space is an integer within {0..K-1}.
        observation_space: A multidiscrete that represents all possible values
            of the state; i.e. all possible values for the amount of cars at 
            each node and all possible nodes for any request.
    """

    def __init__(self, config=env_configs.rideshare_graph_default_config):
        """Inits RideshareGraphEnvironment with the given configuration.

        Args:
            config:
                A dictionary containing the initial configuration of the 
                rideshare graph environment.
        """
        self.config = config
        self.epLen = config['epLen']
        self.graph = nx.Graph(config['edges'])
        self.num_nodes = self.graph.number_of_nodes()
        self.timestep = 0
        self.num_cars = config['num_cars']
        self.lengths = self.find_lengths(self.graph, self.num_nodes)
        self.request_dist = config['request_dist']
        self.reward = config['reward']
        self.reward_denied = config['reward_denied']
        self.reward_fail = config['reward_fail']
        self.cost = config['cost']
        self.fare = config['fare']
        self.max_dist = np.max(self.lengths.flatten())
        self.gamma = config['gamma']
        self.d_threshold = config['d_threshold']
        self.action_space = spaces.Discrete(self.num_nodes)
        vec = [self.num_cars+1 for _ in range(
            self.num_nodes)] + [self.num_nodes, self.num_nodes]
        self.observation_space = spaces.MultiDiscrete(vec)
        self.starting_state = np.asarray(np.concatenate(
            (self.config['starting_state'], self.request_dist(0, self.num_nodes))))
        self.state = self.starting_state

    def reset(self):
        """Reinitializes variables and returns the starting state."""
        self.timestep = 0
        self.state = self.starting_state
        return self.state

    def get_config(self):
        """Returns the configuration for the current environment."""
        return self.config

    def fulfill_req(self, state, dispatch, sink):
        """Update the state to represent a car moving from source to sink.

        Args:
            dispatch:
                An integer representing the dispatched node for the rideshare 
                request.
            sink:
                An integer representing the destination node of the rideshare
                request.
        """
        state[dispatch] -= 1
        state[sink] += 1

    def find_lengths(self, graph, num_nodes):
        """Find the lengths between each pair of nodes in [graph].

        Given a graph, find_lengths first calculates the pairwise shortest distance 
        between all the nodes, which is stored in a (symmetric) matrix.

        Args:
            graph:
                An object containing nodes and edges; each edge has a travel 
                time.
            num_nodes:
                An integer representing the number of nodes in the graph.

        Returns:
            A 2-dimensional symmetric array containing the distances between
            each pair of nodes.
        """
        dict_lengths = dict(nx.all_pairs_dijkstra_path_length(
            graph, cutoff=None, weight='travel_time'))
        lengths = np.zeros((num_nodes, num_nodes))

        for node1 in range(num_nodes):
            for node2 in range(num_nodes):
                lengths[node1, node2] = dict_lengths[node1][node2]
        return lengths

    def step(self, action):
        """Move one step in the environment.

        Args:
            action:
                An Integer representing the node selected by the agent to 
                service the request.

        Returns: A 3-tuple consisting of the following elements:

           - An updated representation of the state, including updated car locations resulting from the previous dispatch and a new ride request,

           - An integer reward value based on the action,

           - A boolean indicating whether or not the model has reached the limit timestep.
        """

        assert self.action_space.contains(action)

        done = False
        accepted = False
        source = self.state[-2]
        sink = self.state[-1]
        newState = np.copy(self.state)
        dispatch_dist = self.lengths[action, source]
        service_dist = self.lengths[source, sink]

        reward_range = self.reward(
            self.fare, self.cost, 0, self.max_dist) - self.reward_fail(self.max_dist, self.cost)
        max_fail_reward = self.reward_fail(self.max_dist, self.cost)

        # If there is a car at the location the agent chose
        if newState[action] > 0:
            exp = np.exp(self.gamma*(dispatch_dist-self.d_threshold))
            prob = 1 / (1 + exp)
            accept = np.random.binomial(1, prob)
            # print("prob: " + str(prob))
            # print("accept: " + str(accept))
            if accept == 1:
                # print('accept service')
                self.fulfill_req(newState, action, sink)
                reward = self.reward(self.fare, self.cost,
                                     dispatch_dist, service_dist)
                reward = (reward - max_fail_reward) / reward_range
                accepted = True
            else:
                # print('decline service')
                reward = (self.reward_denied() -
                          max_fail_reward) / reward_range
        else:
            reward = 0
            done = False

        # updating the state with a new rideshare request
        new_request = self.request_dist(self.timestep, self.num_nodes)
        newState[-2] = new_request[0]
        newState[-1] = new_request[1]
        self.state = newState
        if self.timestep == self.epLen - 1:
            done = True

        self.timestep += 1

        return self.state, np.float64(reward), done, {'request': new_request, 'acceptance': accepted}
