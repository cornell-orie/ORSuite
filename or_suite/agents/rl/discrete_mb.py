import numpy as np
from .. import Agent
import itertools


class DiscreteMB(Agent):

    """
    Uniform Discretization model-based algorithm algorithm  implemented for enviroments
    with continuous states and actions using the metric induces by the l_inf norm


    Attributes:
        epLen: (int) number of steps per episode
        scaling: (float) scaling parameter for confidence intervals
        action_space: (MultiDiscrete) the action space
        state_space: (MultiDiscrete) the state space
        action_net: (list) of a discretization of action space
        state_net: (list) of a discretization of the state space
        alpha: (float) parameter for prior on transition kernel
        flag: (bool) for whether to do full step updates or not
    """

    def __init__(self, action_space, state_space, epLen, scaling, alpha, flag):

        self.epLen = epLen
        self.scaling = scaling
        self.alpha = alpha
        self.flag = flag
        # TODO: Get actual state and action spaces
        self.action_space = action_space
        self.state_space = state_space

        # sizes of action and state spaces
        self.action_size = self.action_space.nvec
        self.state_size = self.state_space.nvec

        # Matrix of size h*S*A
        self.qVals = np.ones([self.epLen] + self.state_size +
                             self.action_size, dtype=np.float32) * self.epLen
        # matrix of size h*S*A
        self.num_visits = np.zeros(
            [self.epLen] + self.state_size + self.action_size, dtype=np.float32)
        print(self.num_visits)
        # matrix of size h*S
        self.vVals = np.ones([self.epLen] + self.state_size,
                             dtype=np.float32) * self.epLen
        # matrix of size h*S*A
        self.rEst = np.zeros([self.epLen] + self.state_size +
                             self.action_size, dtype=np.float32)

        # matrix of size h*S*A*S
        self.pEst = np.zeros([self.epLen] + self.state_size + self.action_size+self.state_size,
                             dtype=np.float32)

        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''

    def reset(self):  # TODO: reset to the way you initialize them
        self.qVals = np.ones([self.epLen] + self.state_size +
                             self.action_size, dtype=np.float32) * self.epLen
        self.vVals = np.ones([self.epLen] + self.state_size,
                             dtype=np.float32) * self.epLen
        self.rEst = np.zeros([self.epLen] + self.state_size +
                             self.action_size, dtype=np.float32)
        self.num_visits = np.zeros(
            [self.epLen] + self.state_size + self.action_size, dtype=np.float32)
        self.pEst = np.zeros([self.epLen] + self.state_size + self.action_size+self.state_size,
                             dtype=np.float32)

    def update_parameters(self, param):
        self.scaling = param

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''

        self.num_visits[timestep, obs, action] += 1

        self.pEst[timestep, obs, action, newObs] += 1

        # timestep, obs, action, newObs

        t = self.num_visits[timestep, obs, action]

        self.rEst[timestep, obs, action] = (
            (t - 1) * self.rEst[timestep, obs, action] + reward) / t

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        # Update value estimates
        if self.flag:  # update estimates via full step updates
            for h in np.arange(self.epLen - 1, -1, -1):
                for state in itertools.product(*[np.arange(self.state_size[0]) for _ in range(self.state_size[0])]):
                    for action in itertools.product(*[np.arange(self.action_size[0]) for _ in range(self.action_size[0])]):
                        dim = (h,) + state + action
                        if self.num_visits[dim] == 0:
                            self.qVals[dim] = self.epLen
                        else:
                            if h == self.epLen - 1:
                                self.qVals[dim] = min(
                                    self.qVals[dim], self.rEst[dim] + self.scaling / np.sqrt(self.num_visits[dim]))
                            else:
                                vEst = min(self.epLen, np.sum(np.multiply(self.vVals[(
                                    h+1,)], self.pEst[dim] + self.alpha) / (np.sum(self.pEst[dim] + self.alpha))))
                                self.qVals[dim] = min(
                                    self.qVals[dim], self.epLen, self.rEst[dim] + self.scaling / np.sqrt(self.num_visits[dim]) + vEst)
                    self.vVals[(h,) + state] = min(self.epLen,
                                                   self.qVals[(h,) + state].max())

    def pick_action(self, state, step):
        '''
        Select action according to a greedy policy

        Args:
            state: int - current state
            timestep: int - timestep *within* episode

        Returns:
            int: action
        '''

        if self.flag == False:  # updates estimates via one step update
            # state_discrete = np.argmin(
            #     (np.abs(np.asarray(self.state_net) - np.asarray(state))), axis=0)
            for action in itertools.product(*[np.arange(self.action_size[0]) for _ in range(self.action_size[0])]):
                dim = (step,) + tuple(state) + action
                if self.num_visits[dim] == 0:
                    self.qVals[dim] == 0
                else:
                    if step == self.epLen - 1:
                        self.qVals[dim] = min(
                            self.qVals[dim], self.rEst[dim] + self.scaling / np.sqrt(self.num_visits[dim]))
                    else:
                        vEst = min(self.epLen, np.sum(np.multiply(self.vVals[(
                            step+1,)], self.pEst[dim] + self.alpha) / (np.sum(self.pEst[dim] + self.alpha))))
                        self.qVals[dim] = min(
                            self.qVals[dim], self.epLen, self.rEst[dim] + self.scaling / np.sqrt(self.num_visits[dim]) + vEst)

            self.vVals[(step,)+tuple(state)] = min(self.epLen,
                                                   self.qVals[(step,) + tuple(state)].max())

        # state_discrete = np.argmin(
        #     (np.abs(np.asarray(self.state_net) - np.asarray(state))), axis=0)
        qFn = self.qVals[(step,)+tuple(state)]
        action = np.asarray(np.where(qFn == qFn.max()))

        index = np.random.choice(len(action[0]))

        actions = ()
        for val in action.T[index]:
            actions += (self.action_space[:, 0][val],)
        return np.asarray(actions)
