import gym
import numpy as np
import sys
from scipy.stats import poisson

from .. import env_configs
import pytest
from stable_baselines3.common.env_checker import check_env

# Finite Armed Bandit Tests
CONFIG = env_configs.rideshare_graph_default_config
np.random.seed(10)

env = gym.make('Rideshare-v0', config=CONFIG)


def test_initial_state():
    for i in range(len(env.starting_state)):
        assert type(
            env.starting_state[i]) == np.int64, "Starting state array does not have all float values"

    # Test to see if timestep starts at zero
    assert env.timestep == 0, "Timestep does not start at 0"

    # Testing if starting state is part of observation space
    assert env.observation_space.contains(
        env.state), "Starting state is not present in given observation space"


def test_step():
    env.reset()
    newState, reward, done, info = env.step(1)

    # Test if new state is part of observation space
    assert env.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    # Test to see if returned reward is a float
    assert type(reward) == np.float64, "Reward is not a float"

    # Check value of reward
    assert reward <= 1 and reward >= 0.0

    # Do step again
    newState, reward, done, info = env.step(1)

    # Test if new state is part of observation space
    assert env.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    # Check value of reward
    assert reward <= 1 and reward >= 0.0

    check_env(env, skip_render_check=True)


def test_bad_action():
    # Testing to see if action not in action space raises an exception
    with pytest.raises(AssertionError):
        env.step(
            [])


def test_reset():
    env.reset()
    assert env.timestep == 0, "Timestep not set to 0 on reset"
    for i in range(len(env.starting_state)):
        assert env.state[i] == env.starting_state[i], "State not set back to starting state on reset at index {}".format(
            i)
