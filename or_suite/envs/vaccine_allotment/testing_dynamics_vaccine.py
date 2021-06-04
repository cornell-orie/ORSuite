#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:33:51 2021

@author: mayleencortez
"""
import numpy as np
from dynamics_model_4groups import dynamics_model

master_seed = 1

# default parameters 1
default_parameters1 = {'contact_matrix': np.array([[0.0001, 0.0001, 0.00003, 0.00003, 0, 0.0001],
                                                   [0, 0.0001, 0.00005, 0.0001, 0, 0],
                                                   [0, 0, 0.00003, 0.00003, 0, 0],
                                                   [0, 0, 0, 0.00003, 0, 0]]),
                       'P': np.array([0.15, 0.15, 0.7, 0.2]),
                       'H': np.array([0.2, 0.2, 0.8, 0.3]),
                       'beta': 1/7,
                       'gamma': 100,
                       'vaccines': 500,
                       'priority': ["1", "2", "3", "4"],
                       'time_step': 7}
starting_state1 = np.array([1090, 2490, 990, 5390, 10, 10, 10, 10, 0, 0, 0])

# default parameters 2
default_parameters2 = {'contact_matrix': np.array([[0.0001, 0.0001, 0.00003, 0.00003, 0, 0.0001],
                                                   [0, 0.0001, 0.00005, 0.0001, 0, 0],
                                                   [0, 0, 0.00003, 0.00003, 0, 0],
                                                   [0, 0, 0, 0.00003, 0, 0]]),
                       'P': np.array([0.15, 0.15, 0.7, 0.2]),
                       'H': np.array([0.2, 0.2, 0.8, 0.3]),
                       'beta': 1/7,
                       'gamma': 100,
                       'vaccines': 500,
                       'priority': [],
                       'time_step': 3}
starting_state2 = np.array([1090, 2490, 990, 5390, 10, 10, 10, 10, 0, 0, 0])

# default parameters 3
default_parameters3 = {'contact_matrix': np.array([[0.0001, 0.0001, 0.00003, 0.00003, 0, 0.0001],
                                                   [0, 0.0001, 0.00005, 0.0001, 0, 0],
                                                   [0, 0, 0.00003, 0.00003, 0, 0],
                                                   [0, 0, 0, 0.00003, 0, 0]]),
                       'P': np.array([0.15, 0.15, 0.7, 0.2]),
                       'H': np.array([0.2, 0.2, 0.8, 0.3]),
                       'beta': 0,
                       'gamma': 100,
                       'vaccines': 500,
                       'priority': ["1", "2", "3", "4"],
                       'time_step': 1}
starting_state3 = np.array([990, 1990, 990, 5990, 10, 10, 10, 10, 0, 0, 0])

np.random.seed(master_seed)
newState, info = dynamics_model(params=default_parameters2, population=starting_state2)
#print(info.keys())
print(newState)
