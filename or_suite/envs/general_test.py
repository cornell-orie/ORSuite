import gym
import numpy as np
import sys
from scipy.stats import poisson

import env_configs
import pytest

from stable_baselines3.common.env_checker import check_env
import general_test_helpers


def test_ambulance_metric():
    general_test_helpers.test_env(
        'Ambulance-v0', env_configs.ambulance_metric_default_config)


def test_ambulance_graph():
    general_test_helpers.test_env(
        'Ambulance-v1', env_configs.ambulance_graph_default_config)


def test_resource():
    general_test_helpers.test_env(
        'Resource-v0', env_configs.resource_allocation_default_config)


def test_bandit():
    general_test_helpers.test_env(
        'Bandit-v0', env_configs.finite_bandit_default_config)


def test_vaccine():
    general_test_helpers.test_env(
        'Vaccine-v0', env_configs.vaccine_default_config1)


def test_rideshare():
    general_test_helpers.test_env(
        'Rideshare-v0', env_configs.rideshare_graph_default_config)


def test_oil():
    general_test_helpers.test_env(
        'Oil-v0', env_configs.oil_environment_default_config)
