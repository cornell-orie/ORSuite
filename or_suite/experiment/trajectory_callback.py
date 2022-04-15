import time
from shutil import copyfile
import pandas as pd
import tracemalloc
import numpy as np
import pickle
import os
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class TrajectoryCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.

    Args:
      print: (bool) Print trajectory info to the console
    """

    def __init__(self, verbose=0):
        self.verbose = verbose
        super(TrajectoryCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        action = self.locals["actions"]
        info = self.locals["infos"]
        next_state = self.locals["new_obs"]
        if self.verbose:
            print('Reward: {}'.format(reward))
            print('Action: {}'.format(action))
            print('Next State: {}'.format(next_state))
            print('Info: {}'.format(info))
        return True
