import numpy as np
import networkx as nx
import sys
from .. import Agent


class maxWeightFixedAgent(Agent):
    def __init__(self, epLen, env_config, alpha):
        """
        Args:
            epLen: number of steps
            func: function used to decide action
            env_config: parameters used in initialization of environment
            data: all data observed so far
        """
        self.data = []
        self.epLen = epLen
        self.num_cars = env_config['num_cars']
        self.alpha = alpha
        self.num_nodes = len(env_config['starting_state'])
        self.graph = nx.Graph(env_config['edges'])
        self.num_nodes = self.graph.number_of_nodes()
        self.lengths = self.find_lengths(self.graph, self.num_nodes)
        self.gamma = env_config['gamma']
        self.d_threshold = env_config['d_threshold']

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

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''
        self.data.append(newObs)
        return

    def reset(self):
        self.data = []

    def update_policy(self, h):
        '''Update internal policy based upon records'''
        return

    def greedy(self, state, epsilon=0):
        '''
        Select action according to function
        '''
        dispatch_dist = self.lengths[state[-2]]
        exp = np.exp(self.gamma*(dispatch_dist-self.d_threshold))
        prob = 1 / (1 + exp)
        weight_value = state[:self.num_nodes] * prob * self.alpha
        action = np.argmax(weight_value)

        return action

    def pick_action(self, state, step):
        action = self.greedy(state)
        return action

    def update_parameters(self, param):
        self.alpha = param
