# ORSuite Experiment Guide
ORSuite is a collection of environments, agents, and instrumentation, aimed at providing researchers in computer science and operations research reinforcement learning implementation of various problems and models arising in operations research. These experiments are made up of several componets including:

- importing packages
- specifying environment
- picking agent list
- running experiment/generating figures

This guide will go through how to read and run experiments made by ORSuite. 

## Package Installation
The package installation is the same for all of the ORSuite experiments. The package below imports several algorithms that help run the algorithms that are created in the experiment. 
```
    -> import or_suite
    -> import numpy as np
    -> import copy
    -> import os
    -> from stable_baselines3.common.monitor import Monitor
    -> from stable_baselines3 import PPO
    -> from stable_baselines3.ppo import MlpPolicy
    -> from stable_baselines3.common.env_util import make_vec_env
    -> from stable_baselines3.common.evaluation import evaluate_policy
    -> import pandas as pd
```
## Experimental Parameters

Most of the environments have simillar parameters. The overalpping parameters can be found in the attributes section of `or_suite/experiment/experiment.py`. 

### Overlapping parameters include: 
   - The parameter `epLen`, an int, represents the length of each episode 
   - `nEps` is an int representing the number of episodes. 
   - `numIters`, an int, is the number of iterations. 
   - `seed` allows random numbers to be generated.
   - `dirPath`, a string, is the location where the data files are stored.
   - `deBug`, a bool, prints information to the command line when set true. 
   - `save_trajectory`, a bool, saves the trajectory information of the ambulance when set to true. 
   - `render` renders the algorithm when set to true.
   - `pickle` is a bool that saves the information to a pickle file when set to true.
 
 ### Environmental specific parameters: 
 Below are extra parameters that specified environments have. 
 
 #### Ambulance: 
 - `alpha`, a float controlling the proportional difference between the cost to move ambulances in between calls and the cost to move the ambulance to respond to a call. If `alpha` is 0, there is no cost to move between calls. If `alpha` is one, there is no cost to respond to calls.
 - `num_ambulance`, an int which represents the number of ambulances in the system.
 - `arrival_dist`,a lambda, is the arrival distribution for calls over the space [0,1]. This takes an integer (step) and returns a float between 0 and 1.
 -  starting_state` is a float list containing the starting locations for each ambulance 
 - `state`, an int list, representing the current state of the environment 
 
#### Inventory: 
    - `lead_times`: array of ints representing the lead times of each supplier
    - `demand_dist`: the random number sampled from the given distribution to be used to calculate the demand
    - `supplier_costs`: array of ints representing the costs of each supplier
    - `hold_cost`: The int holding cost.
    - `backorder_cost`: The backorder holding cost.
    - `max_inventory`: The maximum value (int) that can be held in inventory
    - `max_order`: The maximum value (int) that can be ordered from each supplier
    -  `starting_state`: An int list containing enough indices for the sum of all the lead times, plus an additional index for the initial on-hand inventory.
    - `neg_inventory`: A bool that says whether the on-hand inventory can be negative or not.
 

## Agents

We use specifying agents to compare the effectivness of each agent. They all use multiple agents. While many of the environments use the same agents, they each use a different combination of agents. 

## Running The Code

