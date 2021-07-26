import gym
import numpy as np
import sys
from scipy.stats import poisson

from .. import env_configs
import pytest

CONFIG = env_configs.inventory_control_multiple_suppliers_default_config

env = gym.make('MultipleSuppliers-v0', config=CONFIG)

L = CONFIG['L']
sum_L = 0  # Sum of all leadtimes
for x in range(len(L)):
    sum_L += L[x]


def test_initial_state():
    # Testing state is correct length
    assert len(env.state) == sum_L + \
        1, "State array is not the same as the sum of all leading times plus one"

    # Testing that state has all 0s as starting values.
    for i in range(sum_L + 1):
        assert env.state[i] == 0, "State array has not been initialized to all zeros"

    # Test to see if timestep starts at zero
    assert env.timestep == 0, "Timestep does not start at 0"

    # Testing if starting state is part of observation space
    assert env.observation_space.contains(
        env.state), "Starting state is not present in given observation space"


def test_step():
    np.random.seed(10)
    newState, reward, done, info = env.step([1, 15])

    # Test if new state is part of observation space
    assert env.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    # Test to see if returned reward is a float
    assert type(reward) == float, "Reward is not a float"

    assert reward == 0.0

    expected_state = [0, 0, 0, 0, 1, 15, 0]
    for i in range(sum_L + 1):
        assert env.state[i] == expected_state[i], "New state does not match expected state"

    # Do step again
    newState, reward, done, info = env.step([1, 15])

    # Test if new state is part of observation space
    assert env.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    assert reward == -100.0

    expected_state = [0, 0, 0, 1, 1, 15, 4]
    for i in range(sum_L + 1):
        assert env.state[i] == expected_state[i], "New state does not match expected state"


def test_bad_action():
    # Testing to see if action not in action space raises an exception
    with pytest.raises(AssertionError):
        env.step(
            [0, 0, 0])


def test_reset():
    env.reset()
    assert env.timestep == 0, "Timestep not set to 0 on reset"
    for i in range(sum_L + 1):
        assert env.state[i] == env.starting_state[i], "State not set back to starting state on reset"
