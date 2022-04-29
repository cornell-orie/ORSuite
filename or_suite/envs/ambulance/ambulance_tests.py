import gym
import numpy as np
import sys
from scipy.stats import poisson

from .. import env_configs
import pytest
from stable_baselines3.common.env_checker import check_env

# Ambulance Metric Tests
METRIC_CONFIG = env_configs.ambulance_metric_default_config
GRAPH_CONFIG = env_configs.ambulance_graph_default_config

env = gym.make('Ambulance-v0', config=METRIC_CONFIG)
env2 = gym.make('Ambulance-v1', config=GRAPH_CONFIG)


def test_initial_state():
    for i in range(len(env.starting_state)):
        assert type(env.starting_state[i]) == np.float64 or type(
            env.starting_state[i]) == np.float32, "Starting state array does not have all float values"

    # Test to see if timestep starts at zero
    assert env.timestep == 0, "Timestep does not start at 0"

    # Testing if starting state is part of observation space
    assert env.observation_space.contains(
        env.state), "Starting state is not present in given observation space"


def test_step():
    np.random.seed(10)
    env.reset()
    newState, reward, done, info = env.step([0.8])

    # Test if new state is part of observation space
    assert env.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    # Test to see if returned reward is a float
    assert type(reward) == np.float64 or type(
        reward) == np.float32, "Reward is not a float"

    # Check value of reward
    difference = abs(reward - (-0.239874875))
    assert difference <= .000001 and difference >= 0.0

    # Do step again
    newState, reward, done, info = env.step([0.8])

    # Test if new state is part of observation space
    assert env.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    # Check value of reward
    difference = abs(reward - (-0.0675113))
    assert difference <= .000001 and difference >= 0.0

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


# ------------------------------------------------------------------------------
# Ambulance Graph Tests


def test_initial_state_graph():
    for i in range(len(env2.starting_state)):
        assert type(
            env2.starting_state[i]) == int, "Starting state array does not have all float values"

    # Test to see if timestep starts at zero
    assert env2.timestep == 0, "Timestep does not start at 0"

    # Testing if starting state is part of observation space
    assert env2.observation_space.contains(
        env2.state), "Starting state is not present in given observation space"


def test_step_graph():
    np.random.seed(10)
    env2.reset()
    newState, reward, done, info = env2.step([2, 1])

    # Test if new state is part of observation space
    assert env2.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    # Test to see if returned reward is a float
    assert type(reward) == float, "Reward is not a float"

    # Check value of reward
    difference = abs(reward - (-1.25))
    assert difference <= .000001 and difference >= 0.0

    # Do step again
    newState, reward, done, info = env2.step([2, 1])

    # Test if new state is part of observation space
    assert env2.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    # Check value of reward
    difference = abs(reward - (-1.))
    assert difference <= .000001 and difference >= 0.0

    #check_env(env2, skip_render_check=True)


def test_bad_action_graph():
    # Testing to see if action not in action space raises an exception
    with pytest.raises(AssertionError):
        env2.step(
            [])


def test_reset_graph():
    env2.reset()
    assert env2.timestep == 0, "Timestep not set to 0 on reset"
    for i in range(len(env.starting_state)):
        assert env2.state[i] == env2.starting_state[i], "State not set back to starting state on reset at index {}".format(
            i)
