from sre_parse import State
import numpy as np
from .. import Agent

''' epsilon Net agent '''


class DiscreteQl(Agent):
    """
    Q-Learning algorithm  implemented for enviroments with discrete states and
    actions using the metric induces by the l_inf norm

    TODO: Documentation
    
    Attributes:
        epLen: (int) number of steps per episode
        scaling: (float) scaling parameter for confidence intervals
        action_net: (list) of a discretization of action space
        state_net: (list) of a discretization of the state space
        state_action_dim: d_1 + d_2 dimensions of state and action space respectively
    """

    def __init__(self, action_space, observation_space, epLen, scaling):

        self.state_space = observation_space
        self.action_space = action_space
        self.epLen = epLen
        self.scaling = scaling

        # starts calculating total dimension for the matrix of estimates of Q Values
        dim = np.concatenate((
            np.array([self.epLen]), self.state_space.nvec, self.action_space.nvec))
        self.matrix_dim = dim
        self.qVals = self.epLen * np.ones(self.matrix_dim, dtype=np.float32) # TODO: Initialize with upper bound on max reward via H*max_one_step_reward
                                                                # might need to normalize rewards in your rideshare environment code
                                                                # but otherwise can just use ambulance, that one is already good.
        self.num_visits = np.zeros(self.matrix_dim, dtype=np.float32)

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.environment = env
        pass

        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''

    def update_parameters(self, param):
        self.scaling = param

    def reset(self):
        self.qVals = self.epLen * np.ones(self.matrix_dim, dtype=np.float32)
        self.num_visits = np.zeros(self.matrix_dim, dtype=np.float32)

        '''
            Adds the observation to records by using the update formula
        '''

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''

        self.num_visits[timestep, obs, action] += 1
        t = self.num_visits[timestep, obs, action]
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)

        if timestep == self.epLen-1:
            vFn = 0
        else:
            # vFn = np.max(self.qVals[timestep+1, obs, action]) # nopte this is wrong.
            vFn = np.max(self.qVals[timestep+1, newObs, :])
        vFn = min(self.epLen, vFn)

        self.qVals[timestep, obs, action] = (1 - lr) * self.qVals[timestep, obs, action] + \
            lr * (reward + vFn + bonus)

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        pass

    def pick_action(self, state, step):
        '''
        Select action according to a greedy policy

        Args:
            state: int - current state
            timestep: int - timestep *within* episode

        Returns:
            int: action
        '''
        # returns the state location and takes action based on
        # maximum q value

        # TODO: Add documentation here for this
        a = np.append([step], state)
        qFn = self.qVals[tuple(a)]
        action = np.asarray(np.where(qFn == qFn.max()))
        print(action)
        index = np.random.choice(len(action[0]))
        action = action[0, index]
        action = [action]
        return action
