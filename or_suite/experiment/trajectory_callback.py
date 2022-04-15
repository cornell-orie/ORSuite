import time
from shutil import copyfile
import pandas as pd
import tracemalloc
import numpy as np
import pickle
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
#from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class TrajectoryCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.

    Args:
      print: (bool) Print trajectory info to the console
    """

    def __init__(self,  log_dir: str,  verbose=0):
        self.log_dir = log_dir
        self.verbose = verbose
        super(TrajectoryCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        x, y = ts2xy(load_results(self.log_dir), 'episodes')
        if len(x) > 0:
            reward = y[-1]
            if self.verbose:
                print('Reward: {}'.format(reward))
        return True
