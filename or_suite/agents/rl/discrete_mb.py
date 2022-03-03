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
        self.qVals = np.ones(np.append(np.array([self.epLen]), (self.state_size,
                             self.action_size)), dtype=np.float32) * self.epLen
        # matrix of size h*S*A
        self.num_visits = np.zeros(np.append(np.array([self.epLen]), (self.state_size,
                                                                      self.action_size)), dtype=np.float32)

        # matrix of size h*S
        self.vVals = np.ones(np.append(np.array([self.epLen]), self.state_size),
                             dtype=np.float32) * self.epLen
        # matrix of size h*S*A
        self.rEst = np.zeros(np.append(np.array([self.epLen]), (self.state_size,
                                                                self.action_size)), dtype=np.float32)

        # matrix of size h*S*A*S
        self.pEst = np.zeros(np.append(np.array([self.epLen]), (self.state_size, self.action_size, self.state_size)),
                             dtype=np.float32)

    def reset(self):  # TODO: reset to the way you initialize them
        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''
        self.qVals = np.ones(np.append(np.array([self.epLen]), (self.state_size,
                                                                self.action_size)), dtype=np.float32) * self.epLen
        self.vVals = np.ones(np.append(np.array([self.epLen]), self.state_size),
                             dtype=np.float32) * self.epLen
        self.rEst = np.zeros(np.append(np.array([self.epLen]), (self.state_size,
                                                                self.action_size)), dtype=np.float32)
        self.num_visits = np.zeros(np.append(np.array([self.epLen]), (self.state_size,
                                                                      self.action_size)), dtype=np.float32)
        self.pEst = np.zeros(np.append(np.array([self.epLen]), (self.state_size, self.action_size, self.state_size)),
                             dtype=np.float32)

    def update_parameters(self, param):
        self.scaling = param

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''

        self.num_visits[np.append(np.append([timestep], obs), action)] += 1

        self.pEst[np.append(
            np.append(np.append([timestep], obs), action), newObs)] += 1

        # timestep, obs, action, newObs

        t = self.num_visits[np.append(np.append([timestep], obs), action)]

        self.rEst[np.append(np.append([timestep], obs), action)] = (
            (t - 1) * self.rEst[np.append(np.append([timestep], obs), action)] + reward) / t

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        # Update value estimates
        if self.flag:  # update estimates via full step updates
            for h in np.arange(self.epLen - 1, -1, -1):
                for state in itertools.product(*[np.arange(self.state_size[0]) for _ in range(self.state_space.shape[0])]):
                    for action in itertools.product(*[np.arange(self.action_size[0]) for _ in range(self.action_space.shape[0])]):
                        dim = np.append(np.append([h], state), action)
                        if self.num_visits[tuple(dim)] == 0:
                            self.qVals[tuple(dim)] = self.epLen
                        else:
                            if h == self.epLen - 1:
                                self.qVals[tuple(dim)] = min(
                                    self.qVals[tuple(dim)], self.rEst[tuple(dim)] + self.scaling / np.sqrt(self.num_visits[tuple(dim)]))
                            else:
                                vEst = min(self.epLen, np.sum(np.multiply(self.vVals[(
                                    h+1,)], self.pEst[tuple(dim)] + self.alpha) / (np.sum(self.pEst[tuple(dim)] + self.alpha))))
                                self.qVals[tuple(dim)] = min(
                                    self.qVals[tuple(dim)], self.epLen, self.rEst[tuple(dim)] + self.scaling / np.sqrt(self.num_visits[tuple(dim)]) + vEst)
                    self.vVals[tuple(np.append([h], state))] = min(self.epLen,
                                                                   self.qVals[tuple(np.append([h], state))].max())

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
            for action in itertools.product(*[np.arange(self.action_size[0]) for _ in range(self.action_space.shape[0])]):
               # dim = (step,) + tuple(state) + action
                # dim = np.append(np.asarray([step]),  np.asarray(
                #     state), np.asarray(action))
                dim = np.append(np.append([step], state), action)
                if self.num_visits[tuple(dim)] == 0:
                    self.qVals[tuple(dim)] == 0
                else:
                    if step == self.epLen - 1:
                        self.qVals[tuple(dim)] = min(
                            self.qVals[tuple(dim)], self.rEst[tuple(dim)] + self.scaling / np.sqrt(self.num_visits[tuple(dim)]))
                    else:
                        vEst = min(self.epLen, np.sum(np.multiply(self.vVals[(
                            step+1,)], self.pEst[tuple(dim)] + self.alpha) / (np.sum(self.pEst[tuple(dim)] + self.alpha))))
                        self.qVals[tuple(dim)] = min(
                            self.qVals[tuple(dim)], self.epLen, self.rEst[tuple(dim)] + self.scaling / np.sqrt(self.num_visits[tuple(dim)]) + vEst)

            self.vVals[tuple(np.append([step], state))] = min(self.epLen,
                                                              self.qVals[tuple(np.append([step], state))].max())

        # state_discrete = np.argmin(
        #     (np.abs(np.asarray(self.state_net) - np.asarray(state))), axis=0)
        qFn = self.qVals[tuple(np.append([step], state))]
        action = np.asarray(np.where(qFn == qFn.max()))

        index = np.random.choice(len(action[0]))
       # print(action.T[index])
        action = action[:len(self.state_size), index]
        # print(action)
        return action
        # actions = ()
        # for val in action.T[index]:
        #     actions += (self.action_space[:, 0][val],)
        # return np.asarray(actions)
