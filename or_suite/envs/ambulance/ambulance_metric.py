"""
Implementation of a basic RL environment for continuous spaces.
Includes three test problems which were used in generating the figures.

An ambulance environment over [0,1].  An agent interacts through the environment
by picking a location to station the ambulance.  Then a patient arrives and the ambulance
most go and serve the arrival, paying a cost of travel.
"""

#import rendering
import pyglet
import time
import numpy as np
import gym
from gym import spaces
import math
from .. import env_configs
from gym.envs.classic_control import rendering
# import pyglet
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
renderdir = os.path.dirname(currentdir)
sys.path.append(renderdir)
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)


# ------------------------------------------------------------------------------


class AmbulanceEnvironment(gym.Env):
    """
    A 1-dimensional reinforcement learning environment in the space X = [0, 1].

    Ambulances are located anywhere in X = [0,1], and at the beginning of each 
    iteration, the agent chooses where to station each ambulance (the action).
    A call arrives, and the nearest ambulance goes to the location of that call.

    Attributes:
      epLen: The (int) number of time steps to run the experiment for.
      arrival_dist: A (lambda) arrival distribution for calls over the space [0,1]; takes an integer (step) and returns a float between 0 and 1.
      alpha: A float controlling proportional difference in cost to move between calls and to respond to a call.
      starting_state: A float list containing the starting locations for each ambulance.
      num_ambulance: The (int) number of ambulances in the environment.
      state: An int list representing the current state of the environment.
      timestep: The (int) timestep the current episode is on.
      viewer: The window (Pyglet window or None) where the environment rendering is being drawn.
      most_recent_action: (float list or None) The most recent action chosen by the agent (used to render the environment).
      action_space: (Gym.spaces Box) Actions must be the length of the number of ambulances, every entry is a float between 0 and 1.
      observation_space: (Gym.spaces Box) The environment state must be the length of the number of ambulances, every entry is a float between 0 and 1.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, config=env_configs.ambulance_metric_default_config):
        """

        Args: 
            config: A (dict) dictionary containing the parameters required to set up a metric ambulance environment.
            epLen: The (int) number of time steps to run the experiment for.
            arrival_dist: A (lambda) arrival distribution for calls over the space [0,1]; takes an integer (step) and returns a float between 0 and 1.
            alpha: A float controlling proportional difference in cost to move between calls and to respond to a call.
            starting_state: A float list containing the starting locations for each ambulance.
            num_ambulance: The (int) number of ambulances in the environment.
            norm: The (int) norm used in the calculations.
        """
        super(AmbulanceEnvironment, self).__init__()

        self.config = config
        self.epLen = config['epLen']
        self.alpha = config['alpha']
        self.starting_state = config['starting_state']
        self.state = np.array(self.starting_state, dtype=np.float32)
        self.timestep = 0
        self.num_ambulance = config['num_ambulance']
        self.arrival_dist = config['arrival_dist']
        self.norm = config['norm']
        # variables used for rendering code
        self.viewer = None
        self.most_recent_action = None

        # The action space is a box with each ambulances location between 0 and 1
        self.action_space = spaces.Box(low=0, high=1,
                                       shape=(self.num_ambulance,), dtype=np.float32)

        # The observation space is a box with each ambulances location between 0 and 1
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.num_ambulance,), dtype=np.float32)

    def reset(self):
        """Reinitializes variables and returns the starting state."""

        self.timestep = 0
        self.state = self.starting_state

        return self.starting_state

    def get_config(self):
        return self.config

    def step(self, action):
        """
        Move one step in the environment.

        Args:
            action: A float list of locations in [0,1] the same length as the number of ambulances, where each entry i in the list corresponds to the chosen location for ambulance i.
        Returns:
            float, float list, bool:
            reward: A float representing the reward based on the action chosen.

            newState: A float list representing the state of the environment after the action and call arrival.

            done: A bool flag indicating the end of the episode.
        """
        if isinstance(action, np.ndarray):
            action = action.astype(np.float32)
        assert self.action_space.contains(action)

        old_state = np.array(self.state)

        # The location of the new arrival is chosen randomly from the arrivals
        # distribution arrival_dist
        new_arrival = self.arrival_dist(self.timestep)

        # Update the state of the system according to the action taken and change
        # the location of the closest ambulance to the call to the call location
        action = np.array(action, dtype=np.float32)
        self.most_recent_action = action

        # The closest ambulance to the call is found using the l-1 distance
        close_index = np.argmin(np.abs(action - new_arrival))

        new_state = action.copy()
        new_state[close_index] = new_arrival

        # print("Old", old_state)
        # print("Action", action)
        # print("Close Index", close_index)
        # print("New Arrival", new_arrival)
        # print("New", new_state)

        # The reward is a linear combination of the distance traveled to the action
        # and the distance traveled to the call
        # alpha controls the tradeoff between cost to travel between arrivals and
        # cost to travel to a call
        # The reward is negated so that maximizing it will minimize the distance

        # print("alpha", self.alpha)

        reward = -1 * ((self.alpha / (self.num_ambulance**(1 / self.norm))) * np.linalg.norm(
            action-self.state, self.norm) + (1 - self.alpha) * np.linalg.norm(action-new_state, self.norm))

        # The info dictionary is used to pass the location of the most recent arrival
        # so it can be used by the agent
        info = {'arrival': new_arrival}

        if self.timestep != self.epLen - 1:
            done = False
        else:
            done = True

        self.state = new_state
        self.timestep += 1

        assert self.observation_space.contains(self.state)

        return self.state, reward,  done, info

    def reset_current_step(self, text, line_x1, line_x2, line_y):
        """Used to render a textbox saying the current timestep."""
        self.viewer.reset()
        self.viewer.text("Current timestep: " + str(self.timestep), line_x1, 0)
        self.viewer.text(text, line_x1, 100)
        self.viewer.line(line_x1, line_x2, line_y,
                         width=2, color=rendering.WHITE)

    def draw_ambulances(self, locations, line_x1, line_x2, line_y, ambulance):
        for loc in locations:
            self.viewer.image(line_x1 + (line_x2 - line_x1)
                              * loc, line_y, ambulance, 0.02)
            # self.viewer.circle(line_x1 + (line_x2 - line_x1) * loc, line_y, radius=5, color=rendering.RED)

    def render(self, mode='human'):
        """Renders the environment using a pyglet window."""
        screen_width = 800
        screen_height = 500
        line_x1 = 50
        line_x2 = screen_width - line_x1
        line_y = 300
        script_dir = os.path.dirname(__file__)
        ambulance = pyglet.image.load(script_dir + '/images/ambulance.jpg')
        call = pyglet.image.load(script_dir + '/images/call.jpg')

        screen1, screen2, screen3 = None, None, None

        if self.viewer is None:
            self.viewer = rendering.PygletWindow(
                screen_width + 50, screen_height + 50)

        if self.most_recent_action is not None:

            self.reset_current_step("Action chosen", line_x1, line_x2, line_y)
            self.draw_ambulances(self.most_recent_action,
                                 line_x1, line_x2, line_y, ambulance)
            screen1 = self.viewer.render(mode)
            time.sleep(2)

            self.reset_current_step("Call arrival", line_x1, line_x2, line_y)
            self.draw_ambulances(self.most_recent_action,
                                 line_x1, line_x2, line_y, ambulance)

            arrival_loc = self.state[np.argmax(
                np.abs(self.state - self.most_recent_action))]
            self.viewer.image(line_x1 + (line_x2 - line_x1)
                              * arrival_loc, line_y, call, 0.02)
        #   self.viewer.circle(line_x1 + (line_x2 - line_x1) * arrival_loc, line_y, radius=5, color=rendering.GREEN)
            screen2 = self.viewer.render(mode)
            time.sleep(2)

        self.reset_current_step("Iteration ending state",
                                line_x1, line_x2, line_y)

        self.draw_ambulances(self.state, line_x1, line_x2, line_y, ambulance)

        screen3 = self.viewer.render(mode)
        time.sleep(2)

        return (screen1, screen2, screen3)

    def close(self):
        """Closes the rendering window."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
