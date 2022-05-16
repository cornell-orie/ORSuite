"""

File containing default configurations for the various environments implemented in ORSuite.

"""
import numpy as np
import pandas as pd
import os
import re
import ast


resource_allocation_default_config = {'K': 2,
                                      'num_rounds': 10,
                                      'weight_matrix': np.array([[1, 2], [.3, 9], [1, 1]]),
                                      'init_budget': lambda: 9*10*np.ones(2),
                                      'type_dist': lambda i: 1+np.random.poisson(size=(3), lam=(1, 2, 3)),
                                      'utility_function': lambda x, theta: np.dot(x, theta),
                                      'from_data': False,
                                      'MAX_VAL': 1000
                                      }

resource_allocation_simple_config = {'K': 1,
                                     'num_rounds': 10,
                                     'weight_matrix': np.array([[1]]),
                                     'init_budget': lambda: np.array([20.]),
                                     'utility_function': lambda x, theta: x,
                                     'type_dist': lambda i: np.array([2]),
                                     'from_data': False,
                                     'MAX_VAL': 1000
                                     }


# config with data from MFP Regular Sites
script_dir = os.path.dirname(__file__)
rel_path = './resource_allocation/resource_data/'
resource_file = open(os.path.join(script_dir, rel_path+'MFP.csv'), "r")
df = pd.read_csv(resource_file)
resource_file.close()

data_weights = df['Average Demand per Visit']
data_stdev = df['StDev(Demand per Visit)']

weights_fbst = np.asarray(
    [[3.9, 3.0, 2.8, 2.7, .1], [3.9, 3.0, .1, 2.7, .1], [3.9, 3.0, 2.8, 2.7, 1.9]])
sum_of_rows = weights_fbst.sum(axis=1)
weights_fbst = weights_fbst / sum_of_rows[:, np.newaxis]
dist_types = np.asarray([.25, .3, 1-.25-.3])


class FoodbankAllocationDistribution(object):
    """Class object to preserve state of randomness 
    within length of episode for resource allocation"""

    def __init__(self, n):
        self.epLen = n
        self.max_n = 70

        self.index = np.zeros(self.epLen)
        self.mean_size = np.zeros((self.epLen, len(dist_types)))
        self.stdev_size = np.zeros((self.epLen, len(dist_types)))
        self.reset_index()

    def reset_index(self):
        self.index = np.random.choice(self.max_n, self.epLen, replace=False)
        self.mean_size = np.asarray(
            [dist_types * data_weights[self.index].to_numpy()[j] for j in range(self.epLen)])
        self.stdev_size = np.asarray(
            [(dist_types**2) * data_stdev[self.index].to_numpy()[j] for j in range(self.epLen)])

    def get_type_distribution(self, i):
        arrival = np.maximum(1, np.random.normal(
            self.mean_size, self.stdev_size))[i]

        if i == -2:
            return self.mean_size, self.stdev_size

        if i == -1:
            self.reset_index()

        return arrival

    def get_budget(self):
        return np.asarray([np.sum(self.mean_size)]*5)


def resource_allocation_foodbank_config(n):
    max_n = 70
    assert n <= max_n

    foodbank_allocation_distribution = FoodbankAllocationDistribution(n)

    foodbank_dictionary = {'K': 5,
                           'num_rounds': n,
                           'weight_matrix': weights_fbst,
                           'init_budget': lambda: foodbank_allocation_distribution.get_budget(),
                           'utility_function': lambda x, theta: np.dot(x, theta),
                           'type_dist': lambda i: foodbank_allocation_distribution.get_type_distribution(i),
                           'from_data': True,
                           'MAX_VAL': 1000
                           }

    return foodbank_dictionary


resource_allocation_simple_poisson_config = {'K': 1,
                                             'num_rounds': 10,
                                             'weight_matrix': np.array([[1]]),
                                             'init_budget': lambda: np.array([15.]),
                                             'utility_function': lambda x, theta: x,
                                             'from_data': False,
                                             'type_dist': lambda i: [1+np.random.poisson(lam=0.5)],
                                             'MAX_VAL': 1000
                                             }

ambulance_metric_default_config = {'epLen': 5,
                                   'arrival_dist': lambda x: np.random.beta(5, 2),
                                   'alpha': 0.25,
                                   'starting_state': np.array([0.0], dtype=np.float32),
                                   'num_ambulance': 1,
                                   'norm': 1
                                   }


script_dir = os.path.dirname(__file__)
rel_path = './ambulance/ithaca_data/'

edges_file = open(os.path.join(script_dir, rel_path+'ithaca.edgelist'), "r")
ithaca_edges = []
for line in edges_file:
    travel_dict = ast.literal_eval(re.search('({.+})', line).group(0))
    split = line.split()
    ithaca_edges.append((int(split[0]), int(split[1]), travel_dict))
edges_file.close()


arrivals_file = open(os.path.join(script_dir, rel_path+'arrivals.txt'), "r")
ithaca_arrivals = arrivals_file.read().splitlines()
ithaca_arrivals = [int(i) for i in ithaca_arrivals]
arrivals_file.close()


def from_data(step, num_nodes, ithaca_arrivals):
    node = ithaca_arrivals[step]
    dist = np.full(num_nodes, 0)
    dist[node] = 1
    return dist


ambulance_graph_ithaca_config = {'epLen': 5,
                                 'arrival_dist': from_data,
                                 'alpha': 0.25,
                                 'from_data': True,
                                 'edges': ithaca_edges,
                                 'starting_state': [1, 2],
                                 'num_ambulance': 2,
                                 'data': ithaca_arrivals
                                 }


ambulance_graph_default_config = {'epLen': 5,
                                  'arrival_dist': lambda step, num_nodes: np.full(num_nodes, 1/num_nodes),
                                  'alpha': 0.25,
                                  'from_data': False,
                                  'edges': [(0, 4, {'travel_time': 7}), (0, 1, {'travel_time': 1}), (1, 2, {'travel_time': 3}), (2, 3, {'travel_time': 5}), (1, 3, {'travel_time': 1}), (1, 4, {'travel_time': 17}), (3, 4, {'travel_time': 3})],
                                  'starting_state': [1, 2], 'num_ambulance': 2
                                  }


finite_bandit_default_config = {'epLen': 5,
                                'arm_means': np.array([.1, .7, .2, 1])
                                }

vaccine_default_config1 = {'epLen': 4,
                           'starting_state': np.array([990, 1990, 990, 5990, 10, 10, 10, 10, 0, 0, 0]),
                           'parameters': {'contact_matrix': np.array([[0.0001, 0.0001, 0.00003, 0.00003, 0, 0.0001],
                                                                      [0, 0.0001, 0.00005,
                                                                          0.0001, 0, 0],
                                                                      [0, 0, 0.00003,
                                                                          0.00003, 0, 0],
                                                                      [0, 0, 0, 0.00003, 0, 0]]),
                                          'P': np.array([0.15, 0.15, 0.7, 0.2]),
                                          'H': np.array([0.2, 0.2, 0.8, 0.3]),
                                          'beta': 1/7,
                                          'gamma': 100,
                                          'vaccines': 500,
                                          'priority': ["1", "2", "3", "4"],
                                          'time_step': 7}
                           }

vaccine_default_config2 = {'epLen': 4,
                           'starting_state': np.array([990, 1990, 990, 5990, 10, 10, 10, 10, 0, 0, 0]),
                           'parameters': {'contact_matrix': np.array([[0.0001, 0.0001, 0.00003, 0.00003, 0, 0.0001],
                                                                      [0, 0.0001, 0.00005,
                                                                          0.0001, 0, 0],
                                                                      [0, 0, 0.00003,
                                                                          0.00003, 0, 0],
                                                                      [0, 0, 0, 0.00003, 0, 0]]),
                                          'P': np.array([0.15, 0.15, 0.7, 0.2]),
                                          'H': np.array([0.2, 0.2, 0.8, 0.3]),
                                          'beta': 1/7,
                                          'gamma': 100,
                                          'vaccines': 500,
                                          'priority': [],
                                          'time_step': 7}
                           }

# Importing the NY dataset
script_dir = os.path.dirname(__file__)
ny_rel_path = './ridesharing/ny_data/'

edges_file = open(os.path.join(script_dir, ny_rel_path+'ny.edgelist.txt'), "r")
ny_edges = []
for line in edges_file:
    travel_dict = ast.literal_eval(re.search('({.+})', line).group(0))
    split = line.split()
    ny_edges.append((int(split[0]), int(
        split[1]), travel_dict))
edges_file.close()

ny_arrivals_file = open(os.path.join(
    script_dir, ny_rel_path+'arrivals.txt'), "r")
ny_arrivals = []
for line in ny_arrivals_file:
    split = line.split()
    ny_arrivals.append((int(split[0]), int(split[1])))
ny_arrivals_file.close()


def from_data_ny(step):
    request = ny_arrivals[step]
    return request


rideshare_graph_ny_config = {
    'epLen': 5,
    'edges': ny_edges,
    'starting_state': [10 for _ in range(63)],
    'num_cars': 630,
    'request_dist': lambda step, ny_arrivals: from_data_ny(step),
    'reward': lambda fare, cost, to_source, to_sink: (fare - cost) * to_sink - cost * to_source,
    'reward_denied': lambda: 0,
    'reward_fail': lambda max_dist, cost: -10000 * cost * max_dist,
    'travel_time': lambda velocity, to_sink: int(to_sink / velocity),
    'fare': 6.385456638089008,
    'cost': 1,
    'velocity': 0.011959030558032642,
    'gamma': 1,
    'd_threshold': 4.700448825133434,
    'travel_time': True
}
rideshare_graph_simple_config = {
    'epLen': 5,
    'edges': [(0, 1, {'travel_time': 100}),
              (0, 2, {'travel_time': 1}),
              (1, 2, {'travel_time': 10})
              ],
    'starting_state': [2, 2, 1],
    'num_cars': 5,
    'request_dist': lambda step, num_nodes: np.random.choice(num_nodes, size=2),
    'reward': lambda fare, cost, to_source, to_sink: (fare - cost) * to_sink - cost * to_source,
    'reward_denied': lambda: 0,
    'reward_fail': lambda max_dist, cost: -10000 * cost * max_dist,
    'travel_time': lambda velocity, to_sink: int(to_sink / velocity),
    'fare': 3,
    'cost': 1,
    'velocity': 3,
    'gamma': 1,
    'd_threshold': 20,
    'travel_time': True
}

rideshare_graph_default_config = {
    'epLen': 5,
    'edges': [(0, 1, {'travel_time': 1}), (0, 2, {'travel_time': 100}),
              (0, 3, {'travel_time': 10}), (1, 2, {'travel_time': 20}),
              (1, 3, {'travel_time': 1}), (2, 3, {'travel_time': 70})],
    'starting_state': [1, 1, 1, 0],
    'num_cars': 3,
    'request_dist': lambda step, num_nodes: np.random.choice(num_nodes, size=2),
    'reward': lambda fare, cost, to_source, to_sink: (fare - cost) * to_sink - cost * to_source,
    'reward_denied': lambda: 0,
    'reward_fail': lambda max_dist, cost: -10000 * cost * max_dist,
    'travel_time': lambda velocity, to_sink: int(to_sink / velocity),
    'fare': 3,
    'cost': 1,
    'velocity': 3,
    'gamma': 1,
    'd_threshold': 20,
    'travel_time': True
}

rideshare_graph_2cities_config = {
    'epLen': 5,
    'edges': [(0, 1, {'travel_time': 10}), (0, 2, {'travel_time': 10}), (0, 3, {'travel_time': 50}),
              (3, 4, {'travel_time': 10}), (3, 5, {'travel_time': 10})],
    'starting_state': [0, 0, 1, 0, 1, 1],
    'num_cars': 3,
    'request_dist': lambda step, num_nodes: np.array([np.random.randint(0, 3), np.random.randint(3, 6)]) if np.random.random() > 1/2
    else np.array([np.random.randint(3, 6), np.random.randint(0, 3)]),
    'reward': lambda fare, cost, to_source, to_sink: (fare - cost) * to_sink - cost * to_source,
    'reward_denied': lambda: 0,
    'reward_fail': lambda max_dist, cost: -10000 * cost * max_dist,
    'travel_time': lambda velocity, to_sink: int(to_sink / velocity),
    'fare': 3,
    'cost': 1,
    'velocity': 20,
    'gamma': 1,
    'd_threshold': 15,
    'travel_time': False
}

rideshare_graph_ring_config = {
    'epLen': 5,
    'edges': [(0, 1, {'travel_time': 10}), (1, 2, {'travel_time': 10}),
              (2, 3, {'travel_time': 10}), (3, 4, {'travel_time': 10}),
              (4, 5, {'travel_time': 10}), (5, 6, {'travel_time': 10}),
              (6, 0, {'travel_time': 10})],
    'starting_state': [1, 1, 2, 2, 2, 1, 1],
    'num_cars': 10,
    'request_dist': lambda step, num_nodes: np.random.choice(num_nodes, size=2),
    'reward': lambda fare, cost, to_source, to_sink: (fare - cost) * to_sink - cost * to_source,
    'reward_denied': lambda: 0,
    'reward_fail': lambda max_dist, cost: -10000 * cost * max_dist,
    'travel_time': lambda velocity, to_sink: int(to_sink / velocity),
    'fare': 3,
    'cost': 1,
    'velocity': 10,
    'gamma': 1,
    'd_threshold': 7,
    'travel_time': True
}

# Helper function for setting the initial car location in ithaca. Puts two car every other node


def starting_node_ithaca(num_cars):
    output = [0 for _ in range(630)]
    for i in range(630):
        if i % 2 == 0:
            output[i] = 2

    return output


rideshare_graph_ithaca_config = {
    'epLen': 5,
    'edges': ithaca_edges,
    'starting_state': starting_node_ithaca(630),
    'num_cars': 630,
    'request_dist': lambda step, num_nodes: np.random.choice(num_nodes, size=2),
    'reward': lambda fare, cost, to_source, to_sink: (fare - cost) * to_sink - cost * to_source,
    'reward_denied': lambda: 0,
    'reward_fail': lambda max_dist, cost: -10000 * cost * max_dist,
    'travel_time': lambda velocity, to_sink: int(to_sink / velocity),
    'fare': 3,
    'cost': 1,
    'velocity': 1/3,
    'gamma': 1,
    'd_threshold': 1
}

rideshare_graph_0_1_rides_config = {
    'epLen': 1000,
    'edges': [(0, 1, {'travel_time': 1}), (0, 2, {'travel_time': 5}),
              (0, 3, {'travel_time': 10}), (1, 2, {'travel_time': 4}),
              (1, 3, {'travel_time': 9}), (2, 3, {'travel_time': 5})],
    'starting_state': [1000, 0, 0, 0],
    'num_cars': 1000,
    'request_dist': lambda step, num_nodes: np.array([1, 1]),
    'reward': lambda fare, cost, to_source, to_sink: (fare - cost) * to_sink - cost * to_source,
    'reward_denied': lambda: 0,
    'reward_fail': lambda max_dist, cost: -10000 * cost * max_dist,
    'travel_time': lambda velocity, to_sink: int(to_sink/velocity),
    'fare': 3,
    'cost': 1,
    'velocity': 1/3,
    'gamma': 0.25,
    'd_threshold': 3,
    'travel_time': True
}

rideshare_graph_0_2_rides_config = {
    'epLen': 1000,
    'edges': [(0, 1, {'travel_time': 1}), (0, 2, {'travel_time': 5}),
              (0, 3, {'travel_time': 10}), (1, 2, {'travel_time': 4}),
              (1, 3, {'travel_time': 9}), (2, 3, {'travel_time': 5})],
    'starting_state': [1000, 0, 0, 0],
    'num_cars': 1000,
    'request_dist': lambda step, num_nodes: np.array([2, 2]),
    'reward': lambda fare, cost, to_source, to_sink: (fare - cost) * to_sink - cost * to_source,
    'reward_denied': lambda: 0,
    'reward_fail': lambda max_dist, cost: -10000 * cost * max_dist,
    'travel_time': lambda velocity, to_sink: int(to_sink/velocity),
    'fare': 3,
    'cost': 1,
    'velocity': 1/3,
    'gamma': 0.25,
    'd_threshold': 3,
    'travel_time': True
}

rideshare_graph_0_3_rides_config = {
    'epLen': 1000,
    'edges': [(0, 1, {'travel_time': 1}), (0, 2, {'travel_time': 5}),
              (0, 3, {'travel_time': 10}), (1, 2, {'travel_time': 4}),
              (1, 3, {'travel_time': 9}), (2, 3, {'travel_time': 5})],
    'starting_state': [1000, 0, 0, 0],
    'num_cars': 1000,
    'request_dist': lambda step, num_nodes: np.array([3, 3]),
    'reward': lambda fare, cost, to_source, to_sink: (fare - cost) * to_sink - cost * to_source,
    'reward_denied': lambda: 0,
    'reward_fail': lambda max_dist, cost: -10000 * cost * max_dist,
    'travel_time': lambda velocity, to_sink: int(to_sink/velocity),
    'fare': 3,
    'cost': 1,
    'velocity': 1/3,
    'gamma': 0.25,
    'd_threshold': 3,
    'travel_time': True
}


oil_environment_default_config = {
    'epLen': 5,
    'dim': 1,
    'starting_state': np.asarray([0], dtype=np.float32),
    'oil_prob': lambda x, a, h: np.exp((-1)*np.sum(np.abs(x-a))),
    'cost_param': 0,
    'noise_variance': lambda x, a, h: 0
}

oil_environment_binary_config = {
    'epLen': 5,
    'dim': 1,
    'starting_state': np.asarray([0], dtype=np.float32),
    'oil_prob': lambda x, a, h: np.exp((-1)*np.sum(np.abs(x-(1/9*h)))),
    'cost_param': 0,
    'noise_variance': lambda x, a, h: 0
}

inventory_control_multiple_suppliers_default_config = {
    'lead_times': [1, 5],
    'demand_dist': lambda x: np.random.poisson(10),
    'supplier_costs': [105, 100],
    'hold_cost': 1,
    'backorder_cost': 19,
    'max_inventory': 1000,
    'max_order': 20,
    'epLen': 500,
    'starting_state': None,
    'neg_inventory': True
}

inventory_control_multiple_suppliers_modified_config = {
    'lead_times': [1, 5],
    'demand_dist': lambda x: np.random.poisson(10),
    'supplier_costs': [15, 5],
    'hold_cost': 1,
    'backorder_cost': 200,
    'max_inventory': 1000,
    'max_order': 20,
    'epLen': 500,
    'starting_state': None,
    'neg_inventory': True
}


epLen = 4
airline_default_config = {
    'epLen': epLen,
    'f': np.asarray([1., 2.]),
    'A': np.transpose(np.asarray([[2., 3., 2.], [3., 0., 1.]])),
    'starting_state': np.asarray([20/3, 4., 4.]),
    'P': np.asarray([[1/3, 1/3] for _ in range(epLen+1)])
}
