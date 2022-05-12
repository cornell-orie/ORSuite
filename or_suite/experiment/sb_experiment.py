import time
from shutil import copyfile
import pandas as pd
import tracemalloc
import numpy as np
import pickle
import os
from stable_baselines3.common.monitor import Monitor
from or_suite.experiment.trajectory_callback import *


class SB_Experiment(object):
    """
    Optional instrumentation for running an experiment.

    Runs a simulation between an arbitrary openAI Gym environment and a STABLE BASELINES ALGORITHM, saving a dataset of (reward, time, space) complexity across each episode,
    and optionally saves trajectory information.

    Attributes:
        seed: random seed set to allow reproducibility
        dirPath: (string) location to store the data files
        nEps: (int) number of episodes for the simulation
        deBug: (bool) boolean, when set to true causes the algorithm to print information to the command line
        env: (openAI env) the environment to run the simulations on
        epLen: (int) the length of each episode
        numIters: (int) the number of iterations of (nEps, epLen) pairs to iterate over with the environment
        save_trajectory: (bool) boolean, when set to true saves the entire trajectory information
        render_flag: (bool) boolean, when set to true renders the simulations
        model: (stable baselines algorithm) an algorithm to run the experiments with
        data: (np.array) an array saving the metrics along the sample paths (rewards, time, space)
        trajectory_data: (list) a list saving the trajectory information
    """

    def __init__(self, env, model, dict):
        '''
        Args:
            env: (openAI env) the environment to run the simulations on
            model: (stable baseilnes algorithm) an algorithm to run the experiments with
            dict: a dictionary containing the arguments to send for the experiment, including:

                - dirPath: (string) location to store the data files

                - nEps: (int) number of episodes for the simulation

                - deBug: (bool) boolean, when set to true causes the algorithm to print information to the command line

                - env: (openAI env) the environment to run the simulations on

                - epLen: (int) the length of each episode

                - numIters: (int) the number of iterations of (nEps, epLen) pairs to iterate over with the environment

                - save_trajectory: (bool) boolean, when set to true saves the entire trajectory information
                            TODO: Feature not implemented

                - render: (bool) boolean, when set to true renders the simulations 
                            TODO: Feature not implemeneted
        '''

        self.seed = dict['seed']
        self.dirPath = dict['dirPath']
        self.deBug = dict['deBug']
        self.nEps = dict['nEps']
        self.env = env
        self.epLen = dict['epLen']
        self.num_iters = dict['numIters']
        self.save_trajectory = dict['saveTrajectory']
        self.render_flag = dict['render']

        self.data = np.zeros([dict['nEps']*self.num_iters, 5])

        self.model = model
        # print('epLen: ' + str(self.epLen))

        # if trajectory should be saved, save it in list and make callback
        if self.save_trajectory:
            self.trajectory = []
            self.callback = TrajectoryCallback(verbose=0)

        np.random.seed(self.seed)

    # Runs the experiment
    def run(self):
        '''
            Runs the simulations between an environment and an algorithm
        '''

        # print('**************************************************')
        # print('Running experiment')
        # print('**************************************************')

        index = 0
        traj_index = 0
        episodes = []
        iterations = []
        rewards = []
        times = []
        memory = []

        # Running an experiment
        print(f'New Experiment Run')
        for i in range(self.num_iters):  # loops over all the iterations
            print(f'Iteration: {i}')
            tracemalloc.start()  # starts timer for memory information

            # learns over all of the episodes
            # if trajectory is to be saved, use callback
            if self.save_trajectory:
                self.model.learn(total_timesteps=self.epLen *
                                 self.nEps, callback=self.callback)
            else:
                self.model.learn(total_timesteps=self.epLen*self.nEps)
            self.callback.update_iter()

            current, _ = tracemalloc.get_traced_memory()  # collects memory information
            tracemalloc.stop()

            # appends data to dataset
            episodes = np.append(episodes, np.arange(0, self.nEps))
            iterations = np.append(iterations, [i for _ in range(self.nEps)])

            memory = np.append(memory, [current for _ in range(self.nEps)])

        # save trajectory info
        self.trajectory = self.callback.trajectory

        # print(self.env.get_episode_rewards())
        # print(len(self.env.get_episode_rewards()))
        # rewards are kept cumulatively so we save it out of the loop
        rewards = np.append(rewards, self.env.get_episode_rewards())

        # Times are calculated cumulatively so need to calculate the per iteration time complexity
        orig_times = [0.] + self.env.get_episode_times()
        times = [orig_times[i] - orig_times[i-1]
                 for i in np.arange(1, len(orig_times))]

        # Combining data in dataframe
        # print(episodes)
        # print(iterations)
        # print(rewards)
        # print(memory)
        # print(np.log(times))

        print(len(episodes), len(iterations), len(
            rewards), len(times), len(memory))

        self.data = pd.DataFrame({'episode': episodes,
                                  'iteration': iterations,
                                  'epReward': rewards,
                                  'time': np.log(times),
                                  'memory': memory})

        # print('**************************************************')
        # print('Experiment complete')
        # print('**************************************************')

    # Saves the data to the file location provided to the algorithm
    def save_data(self):
        '''
            Saves the acquired dataset to the noted location

            Returns: dataframe corresponding to the saved data
        '''

        # print('**************************************************')
        # print('Saving data')
        # print('**************************************************')

        # print(self.data)

        dir_path = self.dirPath

        data_loc = 'data.csv'
        traj_loc = 'trajectory.obj'

        dt = self.data
        dt = dt[(dt.T != 0).any()]

        data_filename = os.path.join(dir_path, data_loc)
        traj_filename = os.path.join(dir_path, traj_loc)

        print('Writing to file ' + dir_path + data_loc)

        if os.path.exists(dir_path):
            dt.to_csv(data_filename, index=False,
                      float_format='%.5f', mode='w')
            if self.save_trajectory:  # saves trajectory to filename
                outfile = open(traj_filename, 'wb')
                pickle.dump(self.trajectory, outfile)
                outfile.close()

        else:
            os.makedirs(dir_path)
            dt.to_csv(data_filename, index=False,
                      float_format='%.5f', mode='w')
            if self.save_trajectory:  # saves trajectory to filename
                outfile = open(traj_filename, 'wb')
                pickle.dump(self.trajectory, outfile)
                outfile.close()

        # print('**************************************************')
        # print('Data save complete')
        # print('**************************************************')

        return dt
