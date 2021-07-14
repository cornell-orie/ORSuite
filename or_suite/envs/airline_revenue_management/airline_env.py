import gym
import numpy as np
import sys
import copy
import math

from .. import env_configs


class AirlineRevenueEnvironment(gym.Env):

    def __init__(self, config=env_configs.airline_default_config):
        l = config['l']
        capVal = config['capVal']

        # hi-lo demand hyperparams
        # self.a = 5
        # self.rho = 1

        # checks if we should end
        self.done = False

        self.m = 2 * l
        # slide 11 from lecture 5 has a typo (should be + instead of -)
        self.n = 2 * (l**2 + l)
        num_iter = int(self.n/2)

        self.timeSteps = 0
        self.capa = [capVal]*self.m
        self.capacities = np.array(self.capa)
        self.demands = np.identity(self.m)
        for i in range(l):
            for j in range(l):
                if i != j:
                    demand_col = np.zeros((self.m, 1))
                    demand_col[2 * i + 1] = 1.0
                    demand_col[2 * j] = 1.0
                    self.demands = np.append(self.demands, demand_col, axis=1)
        self.demands = np.append(self.demands, self.demands, axis=1)
        # lowFares = np.random.randint(15,50,self.n//2)
        # self.revenue = np.append(lowFares, 5*lowFares)
        # print(self.revenue)
        # self.revenue = np.asarray([[1.0,2.0][j] for j in range(2) for i in range(num_iter)])
        # the number of time steps we have to finish within
        self.epoch = config['epoch']
        itineraryDemands = np.random.uniform(0, 1, self.n//2)
        scaleTerm = sum(itineraryDemands)
        self.itinDemds = 0.8*itineraryDemands/scaleTerm
        # demdsWoNoArrival = np.append(0.75*self.itinDemds, 0.25*self.itinDemds)
        # self.probabilities = np.append(demdsWoNoArrival, np.asarray(0.2))
        # print(self.probabilities)

        # for l = 3
        self.revenue = np.array([33, 28, 36, 34, 17, 20, 39, 24, 31, 19,
                                 30, 48, 165, 140, 180, 170, 85, 100,
                                 195, 120, 155, 95, 150, 240])
        self.probabilities = np.array([0.01327884, 0.02244177, 0.07923761,
                                       0.0297121,  0.02654582, 0.08408091,
                                       0.09591975, 0.00671065, 0.08147508,
                                       0.00977341, 0.02966204, 0.121162,
                                       0.00442628, 0.00748059, 0.02641254,
                                       0.00990403, 0.00884861, 0.02802697,
                                       0.03197325, 0.00223688, 0.02715836,
                                       0.0032578,  0.00988735, 0.04038733,
                                       0.2])

        # for l = 5
        # self.revenue = np.array([38,  34,  39,  18,  48,  29,  40,  48,  22,  39,  45,  31,  42,  40,  22,  16,  27,  35, \
        #                         40,  42,  15,  42,  32,  40,  36,  24,  41,  33,  33,  38, 190, 170, 195,  90, 240, 145, \
        #                         200, 240, 110, 195, 225, 155, 210, 200, 110,  80, 135, 175, 200, 210,  75, 210, 160, 200, \
        #                         180, 120, 205, 165, 165, 190])

        # self.probabilities = np.array([0.01302623, 0.00630947, 0.0193087 , 0.03749824, 0.0087251 , 0.02197966, \
        #                                  0.0230311 , 0.0250458 , 0.02696926, 0.03631881, 0.00848936, 0.0169562, \
        #                                  0.01757013, 0.01980117, 0.03372276, 0.00092609, 0.01588487, 0.01056883, \
        #                                  0.02438527, 0.00747704, 0.00655709, 0.01516504, 0.01366724, 0.02056504, \
        #                                  0.03065696, 0.02719751, 0.03476736, 0.03692992, 0.00394042, 0.03655934, \
        #                                  0.00434208, 0.00210316, 0.00643623, 0.01249941, 0.00290837, 0.00732655, \
        #                                  0.00767703, 0.0083486 , 0.00898975, 0.01210627, 0.00282979, 0.00565207, \
        #                                  0.00585671, 0.00660039, 0.01124092, 0.0003087 , 0.00529496, 0.00352294, \
        #                                  0.00812842, 0.00249235, 0.0021857 , 0.00505501, 0.00455575, 0.00685501, \
        #                                  0.01021899, 0.00906584, 0.01158912, 0.01230997, 0.00131347, 0.01218645, \
        #                                  0.2       ])

        # the final entry accounts for the probability of no arrival

        self.state = np.array(self.capa)  # Start at beginning of the chain
        self.action_space = gym.spaces.MultiBinary(self.n)
        self.observation_space = gym.spaces.MultiDiscrete(
            [capVal + 1]*self.m)  # capa + 1 since strictly less than
        self.seed()
        metadata = {'render.modes': ['ansi']}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if not self.done:
            self.timeSteps += 1

            # change probabilities
            # log base a
            # hiFareDelta = math.log(self.a+(self.epoch - self.timeSteps)*self.rho, self.a)
            # lowFareDelta = math.log(self.a+(self.timeSteps - 1)*self.rho, self.a)
            # demdsWoNoArrival = np.append(lowFareDelta*self.itinDemds, hiFareDelta*self.itinDemds)
            # self.probabilities = np.append(demdsWoNoArrival, np.asarray(0.2))

            assert self.action_space.contains(action)

            reward = 0.0

            activity = np.random.choice(
                range(self.n+1), 1, list(self.probabilities))[0]

            bookable = True

            for j in range(self.n):
                dems = self.demands[:, j]
                for i in range(len(dems)):
                    if self.state[i] - dems[i] * action[j] < -1e-5:
                        bookable = False
                    else:
                        pass
            if activity != self.n:
                if action[activity] == 1:
                    if bookable:
                        # print("bookable")
                        dems = self.demands[:, activity]
                        self.state -= dems
                        reward = self.revenue[activity]
                        #reward = np.exp(-self.timeSteps/np.sum(self.state))*revThisRound
                        # print(self.reward)
                    else:
                        pass
                else:
                    pass
            else:
                pass

            # # if we don't have the action correlated with "no customer arriving"
            # if activity != self.n:

            #     if action[activity] == 1:
            #         # activity is what class of customer did arrive
            #         # demands for this class is the activity'th column
            #         dems = self.demands[:,activity]
            #         # check if we can book this customer
            #         bookable = True
            #         # check all demands
            #         for i in range(len(dems)):
            #             # check there is enough seats on flight to meet demand
            #             if self.state[i] - dems[i] < 0:
            #                 bookable = False
            #         if bookable:
            #             #print("-----")
            #             #print(self.state)
            #             #print(dems)
            #             #print("-----")

            #             self.state -= dems
            #             reward = self.revenue[activity]
            #             #reward = np.exp(-self.timeSteps/np.sum(self.state))*revThisRound
            #             #print(self.reward)
            #         else:
            #             pass
            #             #print("demand exceed capacity")
            #     elif action[activity] == 0:
            #         # Do nothing
            #         pass
            #     else:
            #         # error
            #         print("action space not binary")
            # else:
            #     pass
            #     #print("No customer arrives")
            self.done = ((np.sum(self.state) == 0)
                         or (self.timeSteps == self.epoch))

        return self.state, reward, self.done, {}

    def render(self, mode='ansi'):
        outfile = sys.stdout if mode == 'ansi' else super(
            AirlineRevenueEnvironment, self).render(mode=mode)
        outfile.write(np.array2string(self.state))

    def reset(self):
        self.state = np.array(self.capa)
        self.done = False
        self.timeSteps = 0
        return self.state
