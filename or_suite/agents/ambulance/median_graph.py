import numpy as np

import networkx as nx
#import sklearn_extra.cluster

import sys
from .. import Agent


def find_lengths(graph, num_nodes):
    """
    Given a graph, find_lengths first calculates the pairwise shortest distance 
    between all the nodes, which is stored in a (symmetric) matrix.
    """
    dict_lengths = dict(nx.all_pairs_dijkstra_path_length(
        graph, cutoff=None, weight='travel_time'))
    lengths = np.zeros((num_nodes, num_nodes))

    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            lengths[node1, node2] = dict_lengths[node1][node2]

    return lengths


class medianAgent(Agent):
    """
    Agent that implements a median-like heuristic algorithm for the graph ambulance environment

    Methods:
        reset() :Clears data and call_locs which contain data on what has occurred so far in the environment
        update_config() : (UNIMPLEMENTED)
        pick_action(state, step) : Chooses locations for each of the ambulances that minimize the 
            distance they would have travelled to respond to all calls that have occurred in the past

    Attributes:
        epLen: (int) number of time steps to run the experiment for
        data: (int list list) a list of all the states of the environment observed so far
        graph: (networkx Graph) a graph representing the observation space
        num_nodes: (int) the number of nodes in the graph
        num_ambulance: (int) the number of ambulances in the environment
        lengths: (float matrix) symmetric matrix containing the distance between each pair of nodes
        call_locs: (int list) the node locations of all calls observed so far

    """

    def __init__(self, epLen, edges, num_ambulance):
        """
        Args:
            epLen: (int) number of time steps to run the experiment for
            edges: (tuple list) a list of tuples, each tuple corresponds to an edge in the graph. The tuples are of the form (int1, int2, {'travel_time': int3}). int1 and int2 are the two endpoints of the edge, and int3 is the time it takes to travel from one endpoint to the other
            num_ambulance: (int) the number of ambulances in the environment
        """
        self.epLen = epLen
        self.data = []
        self.graph = nx.Graph(edges)
        self.num_nodes = self.graph.number_of_nodes()
        self.num_ambulance = num_ambulance
        self.lengths = find_lengths(self.graph, self.num_nodes)
        self.call_locs = []

    def update_config(self, env, config):
        pass

    def reset(self):
        # Resets data and call_locs arrays to be empty
        self.data = []
        self.call_locs = []

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Adds newObs, the most recently observed state, to data
            adds the most recent call arrival, found in info['arrival'] to call_locs'''

        # Adds the most recent state observed in the environment to data
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
        Chooses locations for each of the ambulances that minimize the 
        distance they would have travelled to respond to all calls that have occurred in the past
        """

        counts = np.bincount(self.call_locs, minlength=self.num_nodes)
        # print(self.lengths)
        # print(counts)
        score = self.lengths @ counts
        action = []
        for _ in range(self.num_ambulance):
            node = np.argmin(score)
            action.append(node)
            score[node] = 99999999
        return action

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
