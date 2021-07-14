import gym
import numpy as np
import sys
from scipy.stats import poisson

from .. import env_configs


class DualSourcingEnvironment(gym.Env):

    def __init__(self, config=env_configs.inventory_control_multiple_suppliers_default_config):

        self.L_r = config['L_r']
        self.L_e = config['L_e']
        self.lam = config['lam']
        self.c_r = config['c_r']
        self.c_e = config['c_e']
        self.c_h = config['c_h']
        self.c_b = config['c_b']
        self.I_MAX = config['I_MAX']
        self.a_MAX = config['a_MAX']

        self.I_offset = self.I_MAX/2

        if config['starting'] == None:
            G_hat = np.random.geometric(1/2)
            I_0 = - np.random.poisson(G_hat*self.lam) + self.I_offset
            self.starting = np.append(np.zeros(self.L_r + self.L_e), [I_0])
        else:
            self.starting = config['starting']

        self.state = np.asarray(self.starting)
        self.action_space = gym.spaces.Discrete(self.a_MAX**2)
        self.observation_space = gym.spaces.MultiDiscrete(
            np.append(np.full(self.L_r + self.L_e, self.a_MAX), [self.I_MAX]))
        self.seed()

        metadata = {'render.modes': ['human']}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def demand(self):
        return np.random.poisson(self.lam)

    def transition(self, obs, action, demand):

        (q_r, q_e) = divmod(action, self.a_MAX)  # orders placed

        qs_r = obs[0:self.L_r]

        if self.L_e == 0:
            qs_e = [q_e]
        else:
            qs_e = obs[self.L_r:self.L_r+self.L_e]

        I = obs[-1]

        I_next = min(max(I + qs_r[0] + qs_e[0] - demand, 0), self.I_MAX - 1)

        qs_r_next = np.append(qs_r[1:], q_r)

        if self.L_e == 0:
            qs_e_next = []
        else:
            qs_e_next = np.append(qs_e[1:], q_e)

        obs2 = np.append(np.append(qs_r_next, qs_e_next), [I_next])

        return obs2

    def reward(self, obs, action, obs2):

        c_r = self.c_r
        c_e = self.c_e
        c_h = self.c_h
        c_b = self.c_b

        q_r = obs[0]  # orders receieved

        if self.L_e == 0:
            _, q_e = divmod(action, self.a_MAX)
        else:
            q_e = obs[self.L_r]

        I = obs2[-1]  # new inventory

        I_actual = I - self.I_offset

        if I_actual >= 0:
            reward = -c_r * q_r - c_e * q_e - c_h*I_actual
        else:
            reward = -c_r * q_r - c_e * q_e - c_b*(-I_actual)

        return reward

    def step(self, action):
        assert self.action_space.contains(action)

        obs = self.state
        demand = self.demand()
        obs2 = self.transition(obs, action, demand)
        self.state = obs2
        reward = self.reward(obs, action, obs2)

        done = False

        return self.state, reward, done, {}

    def render(self, mode='human'):
        outfile = sys.stdout if mode == 'human' else super(
            DualSourcingEnvironment, self).render(mode=mode)
        outfile.write(np.array2string(self.state)+'\n')

    def reset(self):
        self.state = np.asarray(self.starting)
        return self.state
