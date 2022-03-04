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

The agents section of the code specifies the algorithms used in the experiment. These agents are later ran against each other to see which ones are most effective for the simulation. 
Each experiment uses multiple agents. While many of the agents overlap, similar to the parameters each experiment has their own combination. 

Common agents throughout different experiments include: 

ambulance - metric: 
- `SB PPO` is Proximal Policy Optimization. When policy is updated, there is a parameter that “clips” each policy update so that action update does not go too far.
- `Random` implements the randomized RL algorithm, which selects an action uniformly at random from the action space. In particular, the algorithm stores an internal copy of the environment’s action space and samples uniformly at random from it.
- `AdaQL` is an Adaptive Discretization Model-Free Agent, implemented for enviroments with continuous states and actions using the metric induced by the l_inf norm.
- `AdaMB` is an Adaptive Discretizaiton Model-Based Agent, implemented for enviroments with continuous states and actions using the metric induced by the l_inf norm.
- `Unif QL` is an eNet Model-Based Agent, implemented for enviroments with continuous states and actions using the metric induces by the l_inf norm.
- `Unif MB` is a eNet Model-Free Agent, implemented for enviroments with continuous states and actions using the metric induces by the l_inf norm.
- `Stable` is an agent that only moves ambulances when responding to an incoming call and not in between calls. This means the policy $\\pi$ chosen by the agent for any given state $X$ will be $\\pi_h(X) = X$
- `Median` is an agent that takes a list of all past call arrivals sorted by arrival location, and partitions it into $k$ quantiles where $k$ is the number of ambulances. The algorithm then selects the middle data point in each quantile as the locations to station the ambulances. 

## Running The Code and Generating Figures 

After running the "Running Algorithm" section, the experiment will run and the agents/algorithms will show up in a chart. This chart will show all of the agents running against each other, with their reward, time and space. Some environments like the metric ambulance will also show MRT and RTV. 


