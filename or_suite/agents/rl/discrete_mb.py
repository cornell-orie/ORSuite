import numpy as np
from gym import spaces
from .. import Agent
import itertools


class DiscreteMB(Agent):

    """
    Uniform model-based algorithm implemented for MultiDiscrete enviroments
    and actions using the metric induces by the l_inf norm


    Attributes:
        epLen: (int) number of steps per episode
        scaling: (float) scaling parameter for confidence intervals
        action_space: (MultiDiscrete) the action space
        state_space: (MultiDiscrete) the state space
        action_size: (list) representing the size of the action sapce
        state_size: (list) representing the size of the state sapce
        alpha: (float) parameter for prior on transition kernel
        flag: (bool) for whether to do full step updates or not
        matrix_dim: (tuple) a concatenation of epLen, state_size, and action_size used to create the estimate arrays of the appropriate size
        qVals: (list) The Q-value estimates for each episode, state, action tuple
        num_visits: (list) The number of times that each episode, state, action tuple has been visited
        vVals: (list) The value function values for every step, state pair
        rEst: (list) Estimates of the reward for a step, state, action tuple
        pEst: (list) Estimates of the number of times that each step, state, action, new_state tuple is considered
    """

    def __init__(self, action_space, state_space, epLen, scaling, alpha, flag):

        self.epLen = epLen
        self.scaling = scaling
        self.alpha = alpha
        self.flag = flag
        # TODO: Get actual state and action spaces
        if isinstance(action_space, spaces.Discrete):
            self.action_space = spaces.MultiDiscrete(
                nvec=np.array([action_space.n]))
            self.multiAction = False
        else:
            self.action_space = action_space
            self.multiAction = True

        self.state_space = state_space

        # sizes of action and state spaces
        self.action_size = self.action_space.nvec
        self.state_size = self.state_space.nvec

        self.matrix_dim = np.concatenate((
            np.array([self.epLen]), self.state_size, self.action_size))

        # Matrix of size h*S*A
        self.qVals = np.ones(self.matrix_dim, dtype=np.float32) * self.epLen
        # matrix of size h*S*A
        self.num_visits = np.zeros(self.matrix_dim, dtype=np.float32)

        # matrix of size h*S
        self.vVals = np.ones(np.append(np.array([self.epLen]), self.state_size),
                             dtype=np.float32) * self.epLen
        # matrix of size h*S*A
        self.rEst = np.zeros(self.matrix_dim, dtype=np.float32)

        # matrix of size h*S*A*S
        self.pEst = np.zeros(np.concatenate((
            np.array([self.epLen]), self.state_size, self.action_size, self.state_size)),
            dtype=np.float32)
        # print(self.pEst.shape)

    def reset(self):  # TODO: reset to the way you initialize them
        '''
            Resets the agent by overwriting all of the estimates back to initial values
        '''
        self.qVals = np.ones(self.matrix_dim, dtype=np.float32) * self.epLen
        self.vVals = np.ones(np.append(np.array([self.epLen]), self.state_size),
                             dtype=np.float32) * self.epLen
        self.rEst = np.zeros(self.matrix_dim, dtype=np.float32)
        self.num_visits = np.zeros(self.matrix_dim, dtype=np.float32)
        self.pEst = np.zeros(np.concatenate((
            np.array([self.epLen]), self.state_size, self.action_size, self.state_size)),
            dtype=np.float32)

    def update_parameters(self, param):
        """Update the scaling parameter.
        Args:
            param: (int) The new scaling value to use"""
        self.scaling = param

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records

        Args:
            obs: (list) The current state
            action: (list) The action taken 
            reward: (int) The calculated reward
            newObs: (list) The next observed state
            timestep: (int) The current timestep
        '''
        # print(
        #     f'Adding on: {timestep}, state: {obs}, action: {action}, reward: {reward}, newObs: {newObs}')
        dim = tuple(np.append(np.append([timestep], obs), action))
        self.num_visits[dim] += 1

        new_obs_dim = tuple(
            np.append(np.append(np.append([timestep], obs), action), newObs))
        self.pEst[new_obs_dim] += 1

        # timestep, obs, action, newObs

        t = self.num_visits[dim]

        self.rEst[dim] = (
            (t - 1) * self.rEst[dim] + reward) / t

        # print(self.num_visits[dim], self.pEst[dim], self.rEst[dim])

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        # Update value estimates
        if self.flag:  # update estimates via full step updates
            for h in np.arange(self.epLen - 1, -1, -1):
                for state in itertools.product(*[np.arange(self.state_size[i]) for i in range(self.state_space.shape[0])]):
                    for action in itertools.product(*[np.arange(self.action_size[j]) for j in range(self.action_space.shape[0])]):
                        dim = tuple(np.append(np.append([h], state), action))
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
                    self.vVals[tuple(np.append([h], state))] = min(self.epLen,
                                                                   self.qVals[tuple(np.append([h], state))].max())

    def pick_action(self, state, step):
        '''
        Select action according to a greedy policy

        Args:
            state: int - current state
            step: int - timestep *within* episode

        Returns:
            list: action
        '''
        if self.flag == False:  # updates estimates via one step update
            # state_discrete = np.argmin(
            #     (np.abs(np.asarray(self.state_net) - np.asarray(state))), axis=0)
            for action in itertools.product(*[np.arange(self.action_size[i]) for i in range(self.action_space.shape[0])]):
               # dim = (step,) + tuple(state) + action
                # dim = np.append(np.asarray([step]),  np.asarray(
                #     state), np.asarray(action))
                dim = tuple(np.append(np.append([step], state), action))
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

            self.vVals[tuple(np.append([step], state))] = min(self.epLen,
                                                              self.qVals[tuple(np.append([step], state))].max())

        # state_discrete = np.argmin(
        #     (np.abs(np.asarray(self.state_net) - np.asarray(state))), axis=0)
        qFn = self.qVals[tuple(np.append([step], state))]
        action = np.asarray(np.where(qFn == qFn.max()))

        index = np.random.choice(len(action[0]))
       # print(action.T[index])
        action = action[:, index]
        #action = action[:len(self.state_size), index]
        # print(action)

        if not self.multiAction:
            action = action[0]
        return action

        # actions = ()
        # for val in action.T[index]:
        #     actions += (self.action_space[:, 0][val],)
        # return np.asarray(actions)
