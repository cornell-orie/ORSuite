# ORSuite Experiment Guide
<!-- Logo -->
<p align="center">
   <img src="https://raw.githubusercontent.com/cornell-orie/ORSuite/main/ORSuite.svg" width="50%">
</p>

ORSuite is a collection of environments, agents, and instrumentation, aimed at providing researchers in computer science and operations research reinforcement learning implementation of various problems and models arising in operations research. These experiments are made up of several componets including:

- importing packages
- specifying environment
- selecting algorithms
- running experiment/generating figures

This guide will go through how to read and run experiments made by ORSuite. 

## Package Installation
The package installation is the same for all of the ORSuite experiments. The packages below imports several algorithms that help run the algorithms that are created in the experiment. 

   -`import or_suite`
   -`import numpy as np` open source Python library 
   - `import copy` creates a shallow and deep copy of a given object
   - `import os` provides functions for working with operating systems
   - `from stable_baselines3.common.monitor import Monitor`
   - `from stable_baselines3 import PPO`
   - `from stable_baselines3.ppo import MlpPolicy`
   - `from stable_baselines3.common.env_util import make_vec_env`
   - `from stable_baselines3.common.evaluation import evaluate_policy`
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
