import gym
import numpy as np
import sys
from scipy.stats import poisson

import env_configs
import pytest

from stable_baselines3.common.env_checker import check_env


def starting_timestep(env):
    assert env.timestep == 0, "Timestep does not start at 0"


def valid_starting_state(env):
    if not env.observation_space.contains(env.state):
        print(env.state)
        print(env.observation_space)
    assert env.observation_space.contains(
        env.state), "Starting state is not in given observation space"


def test_reset(env):
    env.reset()
    assert env.timestep == 0, "Timestep not set to 0 on reset"


def test_bad_action(env):
    # Test to see if invalid acton raises an error
    with pytest.raises(AssertionError):
        env.step([])


def use_env_checker(env):
    with pytest.raises(None):
        check_env(env, skip_render_check=True)


def test_env(id, input_config):
    env = gym.make(id, config=input_config)
    starting_timestep(env)
    valid_starting_state(env)
    test_reset(env)
    test_bad_action(env)
    check_env(env, skip_render_check=True)
