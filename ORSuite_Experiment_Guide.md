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
The specific configuration of the parameters for each of the environments can be found in `or_suite/envs/env_configs.py`.
In order to make an environment you do `Gym.env('Name', env_config)`. 
 

## Agents

The agents section of the code specifies the algorithms used in the experiment. These agents are later ran against each other to see which ones are most effective for the simulation. 
Each experiment uses multiple agents. While many of the agents overlap, similar to the parameters each experiment has their own combination. 

Common agents throughout different experiments include: 
- `SB PPO` is Proximal Policy Optimization. When policy is updated, there is a parameter that “clips” each policy update so that action update does not go too far.
- `Random` implements the randomized RL algorithm, which selects an action uniformly at random from the action space. In particular, the algorithm stores an internal copy of the environment’s action space and samples uniformly at random from it.
- `AdaQL` is an Adaptive Discretization Model-Free Agent, implemented for enviroments with continuous states and actions using the metric induced by the l_inf norm.
- `AdaMB` is an Adaptive Discretizaiton Model-Based Agent, implemented for enviroments with continuous states and actions using the metric induced by the l_inf norm.
- `Unif QL` is an eNet Model-Based Agent, implemented for enviroments with continuous states and actions using the metric induces by the l_inf norm.
- `Unif MB` is a eNet Model-Free Agent, implemented for enviroments with continuous states and actions using the metric induces by the l_inf norm.

Other agents are further specified within each experiment in "ORSuite/examples". 

## Running The Code and Generating Figures 

After running the "Running Algorithm" section, the experiment will run and the agents/algorithms will show up in a chart. This chart will show all of the agents running against each other, with their reward, time and space. With this information one can see which agents are ideal for their goal. Some environments like the metric ambulance will also show MRT and RTV. 

After running the chart is a "Generating Figures" section, where line and radar plots will appear to show how each agent responds visually. 
