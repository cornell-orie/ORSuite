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

Each of the environments have different paramenters. Most of the experiments have similar attributs, which can be found in the attributes section of `or_suite/experiment/experiment.py`. 


