import numpy as np
import networkx as nx
import copy
from .. import Agent


class closetCarAgent(Agent):
    def __init__(self, epLen, env_config):
        """
        Args:
            epLen: number of steps
            func: function used to decide action
            env_config: parameters used in initialization of environment
            data: all data observed so far
        """
        self.env_config = env_config

        self.num_cars = env_config['num_cars']
        self.num_nodes = len(env_config['starting_state'])
        self.epLen = epLen
        self.data = []
        self.lengths = self.get_lengths()

    def get_lengths(self):
        graph = nx.Graph(self.env_config['edges'])
        num_nodes = graph.number_of_nodes()

        return self.find_lengths(graph, num_nodes)

    def reset(self):
        self.data = []

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.env_config = config
        return

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''
        self.data.append(newObs)
        return

    def update_policy(self, h):
        '''Update internal policy based upon records'''
        return

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

    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''
        visited = set(
            [])  # Set of nodes that was visited to check whether there is a car
        lengths_to_source = copy.deepcopy(self.lengths[state[-2]])
        # Array of lengths from the source to a node
        action = np.argmin(lengths_to_source)  # Search for the closest node
        visited.add(action)

        # When the closest node does not have a car to dispatch, we search for the next closest node until
        # 1) we find a node with a car to dispatch or
        # 2) realize there are no cars to dispatch and choose a random action from the action space
        while(state[action] == 0):
            lengths_to_source[action] = float('inf')
            action = np.argmin(lengths_to_source)
            if action in visited:
                return np.random.choice(self.num_nodes)
            else:
                visited.add(action)

        return action

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
