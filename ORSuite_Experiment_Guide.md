<!-- Logo -->
<p align="center">
   <img src="https://raw.githubusercontent.com/cornell-orie/ORSuite/main/ORSuite.svg" width="50%">
</p>
# ORSuite Experiment Guide

ORSuite is a collection of environments, agents, and instrumentation, aimed at providing researchers in computer science and operations research reinforcement learning implementation of various problems and models arising in operations research. These experiments are made up of several componets including:

- importing packages
- specifying the environment
- selecting algorithms
- running experiment/generating figures

This guide will go through how to read and run experiments made by ORSuite. 

## Package Installation
The package installation is the same for all of the ORSuite experiments. The packages below import several modules that help run the algorithms created in the experiment. 

   - `import or_suite` open source with environments created by ORSuite
   - `import numpy as np` open source Python library that aids in scientific computation
   - `import copy` creates a shallow and deep copy of a given object
   - `import os` provides functions for working with operating systems
   - `from stable_baselines3.common.monitor import Monitor` a monitor wrapper for Gym environments, used to know the episode length, time and other data
   - `from stable_baselines3 import PPO` uses clipping so that after an update, the new policy will not be not too far form the old policy
   - `from stable_baselines3.ppo import MlpPolicy` the policy model used in PPO
   - `from stable_baselines3.common.env_util import make_vec_env` stacks multiple different environemnts into one (vecotrized environemnt)
   - `from stable_baselines3.common.evaluation import evaluate_policy` evaluates the agent and the reward
   - `import pandas as pd` brings pandas data analysis library into current environment 

## Experimental Parameters

### Overlapping parameters include: 
   - `epLen`, an int, represents the length of each episode 
   - `nEps`, an int, represents the number of episodes
   - `numIters`, an int, is the number of iterations
   - `seed` allows random numbers to be generated
   - `dirPath`, a string, is the location where the data files are stored
   - `deBug`, a bool, prints information to the command line when set true 
   - `save_trajectory`, a bool, saves the trajectory information of the ambulance when set to true
   - `render` renders the algorithm when set to true
   - `pickle`, a bool, saves the information to a pickle file when set to true
 
 ### Environmental specific parameters: 
 
Most of the environments have simillar parameters. The overalpping parameters can be found in the attributes section of `or_suite/experiment/experiment.py`. 

The specific configuration of the parameters for each of the environments can be found in `or_suite/envs/env_configs.py`.

In order to make an environment you type `Gym.env('Name', env_config)`. 
 

## Agents

The agents section of the code specifies the algorithms used in the experiment. These agents are later ran against each other to see which ones are most effective for the simulation. 
Each experiment uses multiple agents. While many of the agents overlap, similar to the parameters each experiment has their own combination. 

A common agents throughout different experiments is:
- `Random` implements the randomized RL algorithm, which selects an action uniformly at random from the action space. In particular, the algorithm stores an internal copy of the environment’s action space and samples uniformly at random from it.

Other agents are further specified within each experiment in "ORSuite/examples". 

## Running The Code and Generating Figures 

After running the "Running Algorithm" section, the experiment will run and the agents/algorithms will show up in a chart. This chart will show all of the agents running against each other, with their reward, time and space. With this information one can see which agents are ideal for their goal. Some environments like the metric ambulance will also show MRT and RTV. 

After running the chart is a "Generating Figures" section, where line and radar plots will appear to show how each agent responds visually. 

Example: 
When running each of the algorithms they all start with empty lists for each of the paths. 
```
path_list_line = []
algo_list_line = []
path_list_radar = []
algo_list_radar = []

```
Afterwards there are if/elif/else statements to check to see what the current agent at use is. If the current agent is equal to the one specified, then the code will run the cooresponding algorithm. Checking to see whether the SB PPO agent is present would look like: 

```
    if agent == 'SB PPO':
        or_suite.utils.run_single_sb_algo(mon_env, agents[agent], DEFAULT_SETTINGS)
```

Afterwards, a table of agents with each of their rewards, time, space, and for some environments MRT and RTV appears. 
An example of this table is: 

```
  Algorithm    Reward      Time    Space       MRT       RTV
0    Random -1.671218  6.935870 -5053.90 -0.326093 -0.050874
1    Stable -1.032668  7.530278 -4283.30 -0.285655 -0.064205
2    Median -0.875958  6.638060 -5044.62 -0.212675 -0.043899
3     AdaQL -1.113290  6.449667 -4905.76 -0.265052 -0.041170
4     AdaMB -1.113290  6.590761 -4596.32 -0.265052 -0.041170
5   Unif QL -2.137630  6.591951 -4620.32 -0.430012 -0.089006
6   Unif MB -2.299622  6.210666 -4620.32 -0.454634 -0.091324
```

Once the algorithms are run, the figures are created. Each of the environments will create three line plots and one radar plot to show how the difference in agents. 


### Radar Plot
The radar plot below shows the agents (color coded in the box on the right) with the variables the agents are tested against on each end of the radar plot. 
<!-- Radar -->
<p align="center">
   <img src="https://raw.githubusercontent.com/cornell-orie/ORSuite/main/images/radarplotmetric.jpg" width="50%">
</p>

### Line Plot
The line plots also have all of the agents color coded in a box on the right. The first plot shows the reward of each agent. The second one shows the obersved time used on a log scale, and the third shows the observed usage for each episode. 
<!-- Line -->
<p align="center">
   <img src="https://raw.githubusercontent.com/cornell-orie/ORSuite/main/images/MetricLinePlot.jpg" width="50%">
</p>
