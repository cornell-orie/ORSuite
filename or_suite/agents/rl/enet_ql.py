import numpy as np
from .. import Agent

''' epsilon Net agent '''


class eNetQL(Agent):
    """
    Uniform Discretization Q-Learning algorithm  implemented for enviroments
    with continuous states and actions using the metric induces by the l_inf norm


    Attributes:
        epLen: (int) number of steps per episode
        scaling: (float) scaling parameter for confidence intervals
        action_net: (list) of a discretization of action space
        state_net: (list) of a discretization of the state space
        state_action_dim: d_1 + d_2 dimensions of state and action space respectively
    """

    def __init__(self, action_net, state_net, epLen, scaling, state_action_dim):

        self.state_net = np.resize(
            state_net, (state_action_dim[0], len(state_net))).T
        self.action_net = np.resize(
            action_net, (state_action_dim[1], len(action_net))).T
        self.epLen = epLen
        self.scaling = scaling
        self.state_action_dim = state_action_dim

        # starts calculating total dimension for the matrix of estimates of Q Values
        dim = [self.epLen]
        dim += self.state_action_dim[0] * [len(state_net)]
        dim += self.state_action_dim[1] * [len(action_net)]
        self.matrix_dim = dim
        self.qVals = np.ones(self.matrix_dim, dtype=np.float32) * self.epLen
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
        self.qVals = np.ones(self.matrix_dim, dtype=np.float32) * self.epLen
        self.num_visits = np.zeros(self.matrix_dim, dtype=np.float32)

        '''
            Adds the observation to records by using the update formula
        '''

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''

        # returns the discretized state and action location
        state_discrete = np.argmin(
            (np.abs(self.state_net - np.asarray(obs))), axis=0)
        action_discrete = np.argmin(
            (np.abs(self.action_net - np.asarray(action))), axis=0)
        state_new_discrete = np.argmin(
            (np.abs(self.state_net - np.asarray(newObs))), axis=0)

        dim = (timestep,) + tuple(state_discrete) + tuple(action_discrete)
        self.num_visits[dim] += 1
        t = self.num_visits[dim]
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)

        if timestep == self.epLen-1:
            vFn = 0
        else:
            vFn = np.max(self.qVals[(timestep+1,) + tuple(state_new_discrete)])
        vFn = min(self.epLen, vFn)

        self.qVals[dim] = (1 - lr) * self.qVals[dim] + \
            lr * (reward + vFn + bonus)

    def get_num_arms(self):
        ''' Returns the number of arms'''
        return self.epLen * len(self.state_net)**(self.state_action_dim[0]) * len(self.action_net)**(self.state_action_dim[1])

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
        # returns the discretized state location and takes action based on
        # maximum q value
        state_discrete = np.argmin(
            (np.abs(np.asarray(self.state_net) - np.asarray(state))), axis=0)
        qFn = self.qVals[(step,)+tuple(state_discrete)]
        action = np.asarray(np.where(qFn == qFn.max()))
        a = len(action[0])
        index = np.random.choice(len(action[0]))

        actions = ()
        for val in action.T[index]:
            actions += (self.action_net[:, 0][val],)
        return np.asarray(actions)
