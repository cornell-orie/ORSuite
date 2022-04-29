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
    Custom callback for plotting trajectory information for a StatebleBaselines experiment.

    Args:
        verbose: (int) 1 to print trajectory information to console, 0 not to print

    Attributes:
        verbose: (int) 1 to print information, 0 to not print
        trajectory: (list) A list of dictionaries to hold the trajectory information 
        iteration_number: (int) A running count of which iteration training is on
    """

    def __init__(self, verbose=0):
        self.verbose = verbose
        super(TrajectoryCallback, self).__init__(verbose)
        self.trajectory = []
        self.iteration_number = 0

    def _on_step(self) -> bool:
        """
        After each step of training, saves the trajectory information to a dictionary.

        Trajectory information is iteration, episode, step, oldState, action, reward, newState, info.
        """
        # Get values from local variables
        reward = self.locals["rewards"][0]
        action = self.locals["clipped_actions"][0]
        info = self.locals["infos"][0]
        next_state = self.locals["new_obs"][0]
        # if in initial state, have environment's starting state be the previous state
        prev_state = self.trajectory[-1]["newState"] if not len(
            self.trajectory) == 0 else self.locals["env"].get_attr("starting_state")[0]

        # Total number of steps in an episode
        num_steps = self.locals["n_rollout_steps"]

        # calculating current episode number
        episode = int((len(self.trajectory) -
                       self.iteration_number * self.locals["total_timesteps"]) / num_steps)
        step = self.locals["n_steps"]

        # iteration is done in outside loop for training, so it is not taken from self.locals
        iteration = self.iteration_number

        # create dictionary for this time step
        step_dict = {'iter': iteration,
                     'episode': episode,
                     'step': step,
                     'oldState': prev_state,
                     'action': action,
                     'reward': reward,
                     'newState': next_state,
                     'info': info}
        # append this time step to list of trajectory info for all timesteps
        self.trajectory.append(step_dict)

        # print out information
        if self.verbose:
            # print(self.locals)
            print('Iteration: {}'.format(iteration))
            print('Episode: {}'.format(episode))
            print('Step: {}'.format(step))
            print('Old State: {}'.format(prev_state))
            print('Action: {}'.format(action))
            print('Reward: {}'.format(reward))
            print('New State: {}'.format(next_state))
            print('Info: {}'.format(info))
            print('\n')

        return True

    def update_iter(self):
        """
        Updates the iteration number. 

        This is called by sb_experiment.py to manually increment the iteration number in its training loop.
        """
        self.iteration_number += 1
