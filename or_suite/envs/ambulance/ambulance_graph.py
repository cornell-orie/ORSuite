"""Implementation of an RL environment in a discrete graph space.

An ambulance environment over a simple graph.  An agent interacts through 
the environment by selecting locations for various ambulances over the graph.  Afterwards 
a patient arrives and the ambulance most go and serve the arrival, paying a 
cost to travel.
"""

import numpy as np
import gym
from gym import spaces
import networkx as nx
import math

from .. import env_configs

# ------------------------------------------------------------------------------


class AmbulanceGraphEnvironment(gym.Env):
    """
    A graph of nodes V with edges between the nodes E; each node represents a 
    location where an ambulance could be stationed or a call could come in. The 
    edges between nodes are undirected and have a weight representing the distance 
    between those two nodes.
    The nearest ambulance to a call is determined by computing the shortest path 
    from each ambulance to the call, and choosing the ambulance with the minimum 
    length path. The calls arrive according to a prespecified iid probability 
    distribution that can change over time.

    Attributes:
        epLen: The int number of time steps to run the experiment for.
        arrival_dist: A lambda arrival distribution for calls over the observation space; takes an integer (step) and returns an integer that corresponds to a node in the observation space.
        alpha: A float controlling proportional difference in cost to move between calls and to respond to a call.
        from_data: A bool indicator for whether the arrivals will be read from data or randomly generated.
        arrival_data: An int list only used if from_data is True, this is a list of arrivals, where each arrival corresponds to a node in the observation space.
        episode_num: The (int) current episode number, increments every time the environment is reset.
        graph: A networkx Graph representing the observation space.
        num_nodes: The (int) number of nodes in the graph.
        state: An int list representing the current state of the environment.
        timestep: The (int) timestep the current episode is on.
        lengths: A symmetric float matrix containing the distance between each pair of nodes.
        starting_state: An int list containing the starting locations for each ambulance.
        num_ambulance: The (int) number of ambulances in the environment.
        action_space: (Gym.spaces MultiDiscrete) Actions must be the length of the number of ambulances, every entry is an int corresponding to a node in the graph.
        observation_space: (Gym.spaces MultiDiscrete) The environment state must be the length of the number of ambulances, every entry is an int corresponding to a node in the graph.

    """

    metadata = {'render.modes': ['human']}

    def __init__(self, config=env_configs.ambulance_graph_default_config):
        """
        Args: 
            config: A dictionary (dict) containing the parameters required to set up a metric ambulance environment.
            epLen: The (int) number of time steps to run the experiment for.
            arrival_dist: A (lambda) arrival distribution for calls over the observation space; takes an integer (step) and returns an integer that corresponds to a node in the observation space.
            alpha: A float controlling proportional difference in cost to move between calls and to respond to a call.
            from_data: A bool indicator for whether the arrivals will be read from data or randomly generated.
            data: An int list only needed if from_data is True, this is a list of arrivals, where each arrival corresponds to a node in the observation space.
            edges: A tuple list where each tuple corresponds to an edge in the graph. The tuples are of the form (int1, int2, {'travel_time': int3}). int1 and int2 are the two endpoints of the edge, and int3 is the time it takes to travel from one endpoint to the other.
            starting_state: An int list containing the starting locations for each ambulance.
            num_ambulance: The (int) number of ambulances in the environment.
        """
        super(AmbulanceGraphEnvironment, self).__init__()

        self.config = config
        self.epLen = config['epLen']
        self.alpha = config['alpha']
        self.graph = nx.Graph(config['edges'])
        self.num_nodes = self.graph.number_of_nodes()
        self.starting_state = config['starting_state']
        self.state = self.starting_state
        self.timestep = 0
        self.num_ambulance = config['num_ambulance']
        self.arrival_dist = config['arrival_dist']

        self.from_data = config['from_data']

        self.lengths = self.find_lengths(self.graph, self.num_nodes)

        if self.from_data:
            self.arrival_data = config['data']
            self.episode_num = 0

        # Creates an array stored in space_array the length of the number of ambulances
        # where every entry is the number of nodes in the graph
        num_nodes = self.graph.number_of_nodes()
        space_array = np.full(self.num_ambulance, num_nodes)

        # Creates a space where every ambulance can be located at any of the nodes
        self.action_space = spaces.MultiDiscrete(space_array)

        # The definition of the observation space is the same as the action space
        self.observation_space = spaces.MultiDiscrete(space_array)

    def reset(self):
        """Reinitializes variables and returns the starting state."""
        self.timestep = 0
        self.state = self.starting_state

        if self.from_data:
            self.episode_num += 1

        return np.asarray(self.starting_state)

    def get_config(self):
        return self.config

    def step(self, action):
        """
        Move one step in the environment.

        Args:
            action: An int list of nodes the same length as the number of ambulances,
                where each entry i in the list corresponds to the chosen location for 
                ambulance i.
        Returns:
            float, int, bool:
            reward: A float representing the reward based on the action chosen.

            newState: An int list representing the state of the environment after the action and call arrival.

            done: A bool flag indicating the end of the episode.
        """

        assert self.action_space.contains(action)

        old_state = self.state

        # The location of the new arrival is chosen randomly from among the nodes
        # in the graph according to the arrival distribution
        prob_list = []
        if self.from_data:
            dataset_step = (self.episode_num * self.epLen +
                            self.timestep) % len(self.arrival_data)
            prob_list = self.arrival_dist(
                dataset_step, self.num_nodes, self.arrival_data)
        else:
            prob_list = self.arrival_dist(self.timestep, self.num_nodes)
        new_arrival = np.random.choice(self.num_nodes, p=prob_list)

        # Finds the distance traveled by all the ambulances from the old state to
        # the chosen action, assuming that each ambulance takes the shortest path,
        # which is stored in total_dist_oldstate_to_action
        # Also finds the closest ambulance to the call based on their locations at
        # the end of the action, using shortest paths
        shortest_length = 999999999
        closest_amb_idx = 0
        closest_amb_loc = action[closest_amb_idx]

        total_dist_oldstate_to_action = 0

        for amb_idx in range(len(action)):
            new_length = nx.shortest_path_length(
                self.graph, action[amb_idx], new_arrival, weight='travel_time')

            total_dist_oldstate_to_action += nx.shortest_path_length(
                self.graph, self.state[amb_idx], action[amb_idx], weight='dist')

            if new_length < shortest_length:
                shortest_length = new_length
                closest_amb_idx = amb_idx
                closest_amb_loc = action[closest_amb_idx]
            else:
                continue

        # Update the state of the system according to the action taken and change
        # the location of the closest ambulance to the call to the call location
        newState = np.array(action)
        newState[closest_amb_idx] = new_arrival
        obs = newState

        # The reward is a linear combination of the distance traveled to the action
        # and the distance traveled to the call
        # alpha controls the tradeoff between cost to travel between arrivals and
        # cost to travel to a call
        # The reward is negated so that maximizing it will minimize the distance
        reward = -1 * (self.alpha * total_dist_oldstate_to_action +
                       (1 - self.alpha) * shortest_length)

        # The info dictionary is used to pass the location of the most recent arrival
        # so it can be used by the agent
        info = {'arrival': new_arrival}

        if self.timestep != (self.epLen-1):
            done = False
        else:
            done = True

        self.state = newState
        self.timestep += 1

        assert self.observation_space.contains(self.state)

        return self.state, reward,  done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        pass

    def find_lengths(self, graph, num_nodes):
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
