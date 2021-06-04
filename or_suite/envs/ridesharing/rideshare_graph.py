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
        config: a dictionary containing the initial configuration of the 
            rideshare graph environment
        epLen: an integer representing the total number of time steps
        graph: an object containing nodes and edges; each edge has a travel time
        num_nodes: an integer count of the number of nodes in the graph
        starting_state: a vector representing the initial state of the 
            environment; the first K elements represent the number of cars
            at each node, and the final 2 elements represent the current 
            request that needs to be satisfied, i.e. node i to node j
        state: a vector representing the state of the environment; the first K 
            elements represent the number of cars at each node, and the final 2 
            elements represent the current request that needs to be satisfied, 
            i.e. node i to node j
        timestep: an integer representing the current timestep of the model
        num_cars: an integer representing the number of cars in the model
        lengths: a 2-dimensional symmetric array containing the distances 
            between each pair of nodes
        request_dist: a vector consisting of the distribution used for selecting
            nodes when generating requests
        reward: a lambda function to generate the reward
        reward_fail: a lambda function to generate the reward when the RL
            agent fails; i.e. when a request is not satisfied
        action_space: a discrete set of values the action can have; in this case
            the action space is an integer within {0..K-1}
        observation_space: a multidiscrete that represents all possible values
            of the state; i.e. all possible values for the amount of cars at 
            each node and all possible nodes for any request 
    """


    def __init__(self, config=env_configs.rideshare_graph_default_config):
        """Inits RideshareGraphEnvironment with the given configuration.
        
        Args:
            config:
                a dictionary containing the initial configuration of the 
                rideshare graph environment
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
        self.reward_fail = config['reward_fail']
        self.gamma = config['gamma']
        self.d_threshold = config['d_threshold']
        self.action_space = spaces.Discrete(self.num_nodes)
        vec = [self.num_cars for _ in range(self.num_nodes)] + [self.num_nodes, self.num_nodes]
        self.observation_space = spaces.MultiDiscrete(vec)
        self.starting_state = np.asarray(np.concatenate((self.config['starting_state'], self.request_dist(0, self.num_nodes))))
        self.state = self.starting_state


    def reset(self):
        """Reinitializes variables and returns the starting state."""
        self.timestep = 0
        self.state = np.asarray(np.concatenate((self.config['starting_state'], self.request_dist(0, self.num_nodes))))
        
        return self.state


    def get_config(self):
        """Returns the configuration for the current environment"""
        return self.config


    def fulfill_req(self, dispatch, sink):
        """Update the state to represent a car moving from source to sink
        
        Args:
            dispatch:
                an integer representing the dispatched node for the rideshare 
                request
            sink:
                an integer representing the destination node of the rideshare
                request
        """
        self.state[dispatch] -= 1
        self.state[sink] += 1


    def find_lengths(self, graph, num_nodes):
        """Find the lengths between each pair of nodes in [graph].

        Given a graph, find_lengths first calculates the pairwise shortest distance 
        between all the nodes, which is stored in a (symmetric) matrix.

        Args:
            graph:
                an object containing nodes and edges; each edge has a travel 
                time
            num_nodes:
                an integer representing the number of nodes in the graph

        Returns:
            A 2-dimensional symmetric array containing the distances between
            each pair of nodes
        """
        dict_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='travel_time'))
        lengths = np.zeros((num_nodes, num_nodes))

        for node1 in range(num_nodes):
            for node2 in range(num_nodes):
                lengths[node1, node2] = dict_lengths[node1][node2]
        return lengths


    def step(self, action):
        """Move one step in the environment
        
        Args:
            action:
                an Integer representing the node selected by the agent to 
                service the request

        Returns:
            A 3-tuple consisting of the following elements:
            
            An updated representation of the state, including updated car
            locations resulting from the previous dispatch and a new ride
            request,

            An integer reward value based on the action,

            A boolean indicating whether or not the model has reached the limit
            timestep.
        """
        assert self.action_space.contains(action)

        done = False
        source = self.state[-2]
        sink = self.state[-1]
        dispatch_dist = self.lengths[action, source]

        if self.state[action] > 0:
            exp = np.exp((-1)*self.gamma*(dispatch_dist-self.d_threshold))
            prob = exp / (1 + exp)
            accept = np.random.binomial(1, prob)
            # print("prob: " + str(prob))
            # print("accept: " + str(accept))
            if accept == 1:
                # print('accept service')
                self.fulfill_req(action, sink)
                reward = self.reward(dispatch_dist)
            else:
                # print('decline service')
                reward = self.reward_fail(dispatch_dist)
        else:
            reward = self.reward_fail(dispatch_dist)
            done = True

        # updating the state with a new rideshare request
        new_request = self.request_dist(self.timestep, self.num_nodes)
        self.state[-2] = new_request[0]
        self.state[-1] = new_request[1]
        
        if self.timestep >= self.epLen:
            done = True

        self.timestep += 1
        

        return self.state, reward, done, {'request': new_request}