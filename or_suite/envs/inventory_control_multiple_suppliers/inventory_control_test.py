import gym
import numpy as np
import sys
from scipy.stats import poisson

from .. import env_configs
import pytest
from stable_baselines3.common.env_checker import check_env

# These tests are for 2 suppliers
CONFIG = env_configs.inventory_control_multiple_suppliers_default_config

env = gym.make('MultipleSuppliers-v0', config=CONFIG)

lead_times = CONFIG['lead_times']
sum_L = 0  # Sum of all leadtimes
for x in range(len(lead_times)):
    sum_L += lead_times[x]


CONFIG3 = {'lead_times': [5, 1, 8],
           'demand_dist': lambda x: np.random.poisson(10),
           'supplier_costs': [100, 105, 90],
           'hold_cost': 1,
           'backorder_cost': 19,
           'max_inventory': 1000,
           'max_order': 20,
           'epLen': 500,
           'starting_state': None,
           'neg_inventory': True}

env3 = gym.make('MultipleSuppliers-v0', config=CONFIG3)

L3 = CONFIG3['lead_times']
sum_L3 = 0  # Sum of all leadtimes
for x in range(len(L3)):
    sum_L3 += L3[x]


def test_initial_state():
    # Testing state is correct length
    assert len(env.state) == sum_L + \
        1, "State array is not the same as the sum of all leading times plus one"

    # Testing that state has all 0s as starting values.
    for i in range(sum_L):
        assert env.state[i] == 0, "State array has not been initialized to all zeros"
        assert env.state[-1] == env.max_inventory, "Last index is not max"

    # Test to see if timestep starts at zero
    assert env.timestep == 0, "Timestep does not start at 0"

    # Testing if starting state is part of observation space
    assert env.observation_space.contains(
        env.state), "Starting state is not present in given observation space"


def test_step():
    np.random.seed(10)
    env.reset()
    newState, reward, done, info = env.step([1, 15])

    # Test if new state is part of observation space
    assert env.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    # Test to see if returned reward is a float
    assert type(reward) == float, "Reward is not a float"

    assert reward == -1852.0

    expected_state = [1, 0, 0, 0, 0, 15, 987]
    for i in range(sum_L + 1):
        assert env.state[i] == expected_state[i], "New state does not match expected state at index {}".format(
            i)

    # Do step again
    newState, reward, done, info = env.step([1, 15])

    # Test if new state is part of observation space
    assert env.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    assert reward == -2042.0

    expected_state = [1, 0, 0, 0, 15, 15, 977]
    for i in range(sum_L + 1):
        assert env.state[i] == expected_state[i], "New state does not match expected state at index {}".format(
            i)
    check_env(env, skip_render_check=True)


def test_bad_action():
    # Testing to see if action not in action space raises an exception
    with pytest.raises(AssertionError):
        env.step(
            [0, 0, 0])


def test_reset():
    env.reset()
    assert env.timestep == 0, "Timestep not set to 0 on reset"
    for i in range(sum_L):
        assert env.state[i] == env.starting_state[i], "State not set back to starting state on reset at index {}".format(
            i)
    assert env.state[-1] == env.max_inventory


# These tests are for three suppliers

def test_initial_state_three():
    # Testing state is correct length
    assert len(env3.state) == sum_L3 + \
        1, "State array is not the same as the sum of all leading times plus one"

    # Testing that state has all 0s as starting values.
    for i in range(sum_L3):
        assert env3.state[i] == 0, "State array has not been initialized to all zeros"
        assert env3.state[-1] == env3.max_inventory, "Last index is not max"

    # Test to see if timestep starts at zero
    assert env3.timestep == 0, "Timestep does not start at 0"

    # Testing if starting state is part of observation space
    assert env3.observation_space.contains(
        env3.state), "Starting state is not present in given observation space"


def test_step_three():
    np.random.seed(10)
    env3.reset()
    newState, reward, done, info = env3.step([1, 15, 4])

    # Test if new state is part of observation space
    assert env3.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    # Test to see if returned reward is a float
    assert type(reward) == float, "Reward is not a float"

    assert reward == -2282.0

    expected_state = [0, 0, 0, 0, 1, 15, 0, 0, 0, 0, 0, 0, 0, 4, 987]
    for i in range(sum_L3 + 1):
        assert env3.state[i] == expected_state[i], "New state does not match expected state at index {}".format(
            i)

    # Do step again
    newState, reward, done, info = env3.step([1, 15, 4])

    # Test if new state is part of observation space
    assert env3.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    assert reward == -2206.0

    expected_state = [0, 0, 0, 1, 1, 15, 0, 0, 0, 0, 0, 0, 4, 4, 991]
    for i in range(sum_L3 + 1):
        assert env3.state[i] == expected_state[i], "New state does not match expected state at index {}".format(
            i)
    check_env(env3, skip_render_check=True)


def test_bad_action_three():
    # Testing to see if action not in action space raises an exception
    with pytest.raises(AssertionError):
        env3.step(
            [])


def test_reset_three():
    env3.reset()
    assert env.timestep == 0, "Timestep not set to 0 on reset"
    for i in range(sum_L3):
        assert env3.state[i] == env3.starting_state[i], "State not set back to starting state on reset at index {}".format(
            i)
    assert env3.state[-1] == env3.max_inventory
