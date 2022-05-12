import numpy as np
import cvxpy as cp
import pandas as pd
import or_suite
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.ppo import MlpPolicy

"""
Helper code to run a single simulation of either an ORSuite experiment or the wrapper for a stable baselines algorithm.
"""


def run_single_algo(env, agent, settings):
    """
    Runs a single experiment.

    Args:
        env: The environment.
        agent: The agent.
        setting: A dictionary containing experiment settings.
    """
    exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()


def run_single_algo_tune(env, agent, param_list, settings):
    best_reward = (-1)*np.inf
    best_param = param_list[0]

    for param in param_list:
        agent.reset()
        agent.update_parameters(param)

        exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
        exp.run()
        dt = pd.DataFrame(exp.data, columns=[
                          'episode', 'iteration', 'epReward', 'memory', 'time'])
        avg_end_reward = dt[dt['episode'] ==
                            dt.max()['episode']].iloc[0]['epReward']
        # print(avg_end_reward)
        if avg_end_reward >= best_reward:
            best_reward = avg_end_reward

            best_param = param
            best_exp = exp
    print(f"Chosen parameters: {best_param}")
    best_exp.save_data()
    print(best_param)

# Helper code to run single stable baseline experiment


def run_single_sb_algo(env, agent, settings):
    """
    Runs a single experiment.

    Args:
        env: The environment.
        agent: The agent.
        setting: A dictionary containing experiment settings.
    """

    exp = or_suite.experiment.sb_experiment.SB_Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()


def run_single_sb_algo_tune(env, agent, epLen, param_list, settings):
    best_reward = (-1)*np.inf
    best_param = (param_list['learning_rate'][0], param_list['gamma'][0])

    if agent == 'SB PPO':
        for learning_rate in param_list['learning_rate']:
            for gamma in param_list['gamma']:
                mon_env = Monitor(env)
                model = PPO(MlpPolicy, mon_env, learning_rate=learning_rate, gamma=gamma,
                            verbose=0, n_steps=epLen)
                exp = or_suite.experiment.sb_experiment.SB_Experiment(
                    mon_env, model, settings)
                exp.data = np.zeros([exp.nEps*exp.num_iters, 5])
                exp.run()
                dt = pd.DataFrame(exp.data, columns=[
                    'episode', 'iteration', 'epReward', 'time', 'memory'])
                avg_end_reward = dt[dt['episode'] ==
                                    dt.max()['episode']].iloc[0]['epReward']
                # print(avg_end_reward)
                if avg_end_reward >= best_reward:
                    best_reward = avg_end_reward

                    best_param = (learning_rate, gamma)
                    best_exp = exp
    elif agent == 'SB DQN':
        for learning_rate in param_list['learning_rate']:
            for gamma in param_list['gamma']:
                model = DQN(MlpPolicy, env, learning_rate=learning_rate, gamma=gamma,
                            verbose=0, n_steps=epLen)
                exp = or_suite.experiment.sb_experiment.SB_Experiment(
                    env, model, settings)

                exp.run()
                dt = pd.DataFrame(exp.data, columns=[
                    'episode', 'iteration', 'epReward', 'time', 'memory'])
                avg_end_reward = dt[dt['episode'] ==
                                    dt.max()['episode']].iloc[0]['epReward']
                # print(avg_end_reward)
                if avg_end_reward >= best_reward:
                    best_reward = avg_end_reward

                    best_param = (learning_rate, gamma)
                    best_exp = exp

    print(f"Chosen parameters: {best_param}")
    best_exp.save_data()
    print(best_param)


'''
PROBLEM DEPENDENT METRICS

Sample implementation of problem dependent metrics.  Each one of them should take in a trajectory (as output and saved in an experiment)
and return a corresponding value, where large corresponds to 'good'.
'''


'''
RIDESHARING ENVIRONMENT
'''
# Calculating the acceptance rate for the ridesharing environment on the trajectory datafile


def acceptance_rate(traj, dist):
    accepted = 0
    for i in range(len(traj)):
        cur_data = traj[i]
        if cur_data['info']['acceptance']:
            accepted += 1
    return accepted / len(traj)

# Calculating the mean of the dispatched distance of the ridesharing environment on the trajectory datafile


def mean_dispatch_dist(traj, dist):
    dispatch_dists = 0
    for i in range(len(traj)):
        cur_data = traj[i]
        cur_state = cur_data['oldState']
        dispatch_dists += dist(cur_data['action'], cur_state[-2])
    return (-1) * dispatch_dists / len(traj)

# Calculating the variance of the dispatched distance of the ridesharing environment on the trajectory datafile


def var_dispatch_dist(traj, dist):
    dispatch_dists = []
    for i in range(len(traj)):
        cur_data = traj[i]
        cur_state = cur_data['oldState']
        dispatch_dists.append(dist(cur_data['action'], cur_state[-2]))
    return (-1) * np.var(dispatch_dists)


'''
AMBULANCE ENVIRONMENT
'''

# Calculating mean response time for ambulance environment on the trajectory datafile


def mean_response_time(traj, dist):
    mrt = 0
    for i in range(len(traj)):
        cur_data = traj[i]
        mrt += (-1) * \
            np.min(
                dist(np.array(cur_data['action']), cur_data['info']['arrival']))
    return mrt / len(traj)

# Calculating the variance in the response time for ambulance environment on the trajectory datafile


def response_time_variance(traj, dist):
    dists = []
    for i in range(len(traj)):
        cur_data = traj[i]
        dists.append(
            np.min(dist(np.array(cur_data['action']), cur_data['info']['arrival'])))
    return (-1)*np.var(dists)


'''
RESOURCE ALLOCATION ENVIRONMENT
'''


def offline_opt(budget, size, weights, solver):
    """
    Uses solver from generate_cvxpy_solve and applies it to values.

    Inputs:
        budget: Initial budget for K commodities.
        size: 2D numpy array of sizes of each type at each location.
        weights: 2D numpy array containing the demands of each type.
    """
    tot_size = np.sum(size, axis=0)
    _, x = solver(tot_size, weights, budget)
    allocation = np.zeros((size.shape[0], weights.shape[0], weights.shape[1]))
    for i in range(size.shape[0]):
        allocation[i, :, :] = x
    return allocation


def generate_cvxpy_solve(num_types, num_resources):
    """
    Creates a generic solver to solve the offline resource allocation problem.

    Inputs: 
        num_types: Number of types.
        num_resources: Number of resources.
    Returns:
        prob: CVXPY problem object.
        solver: Function that solves the problem given data.
    """
    x = cp.Variable(shape=(num_types, num_resources))
    sizes = cp.Parameter(num_types, nonneg=True)
    weights = cp.Parameter((num_types, num_resources), nonneg=True)
    budget = cp.Parameter(num_resources, nonneg=True)
    objective = cp.Maximize(
        cp.log(cp.sum(cp.multiply(x, weights), axis=1)) @ sizes)
    constraints = []
    constraints += [0 <= x]
    for i in range(num_resources):
        constraints += [x[:, i] @ sizes <= budget[i]]
    # constraints += [x @ sizes <= budget]
    prob = cp.Problem(objective, constraints)

    def solver(true_sizes, true_weights, true_budget):
        sizes.value = true_sizes
        weights.value = true_weights
        budget.value = true_budget
        prob.solve()
        return prob.value, np.around(x.value, 5)
    return prob, solver


def times_out_of_budget(traj, env_config):
    num_iter = traj[-1]['iter']+1
    num_eps = traj[-1]['episode']+1
    num_steps = traj[-1]['step']+1
    num_types, num_commodities = traj[-1]['action'].shape

    times_out_budget = 0
    traj_index = 0
    # for dict in traj:
    #     print()
    #     for k, v in dict.items():
    #         print(k, v)

    for iter_num in range(num_iter):
        for ep in range(num_eps):
            cur_dict = traj[traj_index]
            budget = cur_dict['oldState'][:num_commodities].copy()
            # budget = np.copy(env_config['init_budget']())
            for step in range(num_steps):
                # print(f"retrieved budget for traj index {traj_index} is {budget} in times_out_of_budget")
                cur_dict = traj[traj_index]
                old_budget = cur_dict['oldState'][:num_commodities].copy()
                old_type = cur_dict['oldState'][num_commodities:].copy()
                allocation = cur_dict['action'].copy()

                if np.min(old_budget - np.matmul(old_type, allocation)) >= -.0005:
                    # updates the budget by the old budget and the allocation given
                    if traj_index != ep - 1:
                        # temp budget in case of rounding errors
                        budget = old_budget-np.matmul(old_type, allocation)

                    else:
                        budget = budget
                else:  # algorithm is allocating more than the budget, output a negative infinity reward
                    budget = old_budget
                    times_out_budget += 1

                traj_index += 1

    return times_out_budget/num_iter


def delta_EFFICIENCY(traj, env_config):
    num_iter = traj[-1]['iter']+1
    num_eps = traj[-1]['episode']+1
    num_steps = traj[-1]['step']+1
    # print('Iters: %s, Eps: %s, Steps: %s'%(num_iter,num_eps,num_steps))
    num_types, num_commodities = traj[-1]['action'].shape
    final_avg_efficiency = np.zeros(num_eps)

    traj_index = 0
    for iter_num in range(num_iter):
        # print(iter_num)
        for ep in range(num_eps):
            # pull out cur_dict for init_bud and curr_arrival
            cur_dict = traj[traj_index]
            budget = cur_dict['oldState'][:num_commodities].copy()
            # print(ep)
            # print('budget:' + str(budget))
            for step in range(num_steps):
                cur_dict = traj[traj_index]
                # print(cur_dict['oldState'])
                # print(cur_dict['action'])
                budget -= np.matmul(cur_dict['oldState']
                                    [num_commodities:], cur_dict['action'])
                traj_index += 1
#                budget -= cur_dict['oldState']
            final_avg_efficiency[ep] += np.sum(budget)
    return (-1)*np.mean(final_avg_efficiency)

    return 0


def delta_PROP(traj, env_config):
    weight_matrix = env_config['weight_matrix']
    utility_function = env_config['utility_function']
    num_iter = traj[-1]['iter']+1
    num_eps = traj[-1]['episode']+1
    num_steps = traj[-1]['step']+1
    num_types, num_commodities = traj[-1]['action'].shape
    final_avg_prop = np.zeros(num_eps)

    traj_index = 0
    for iter_num in range(num_iter):
        for ep in range(num_eps):

            # budget = np.copy(env_config['init_budget']())
            cur_dict = traj[traj_index]
            budget = cur_dict['oldState'][:num_commodities].copy()
            X_alg = np.zeros((num_steps, num_types, num_commodities))
            sizes = np.zeros((num_steps, num_types))

            for step in range(num_steps):
                cur_dict = traj[traj_index]

                X_alg[step] = cur_dict['action'].copy()
                sizes[step] = cur_dict['oldState'][num_commodities:].copy()
                traj_index += 1

            prop_alloc = budget / np.sum(sizes)
            max_prop = 0

            for theta in range(num_types):
                for h in range(num_steps):
                    max_prop = max(max_prop, utility_function(
                        prop_alloc, weight_matrix[theta, :]) - utility_function(X_alg[h, theta], weight_matrix[theta, :]))

            final_avg_prop[ep] += max_prop

    return (-1)*np.mean(final_avg_prop)


def delta_HINDSIGHT_ENVY(traj, env_config):
    weight_matrix = env_config['weight_matrix']
    utility_function = env_config['utility_function']
    num_iter = traj[-1]['iter']+1
    num_eps = traj[-1]['episode']+1
    num_steps = traj[-1]['step']+1
    num_types, num_commodities = traj[-1]['action'].shape
    final_avg_envy = np.zeros(num_eps)

    traj_index = 0
    for iter_num in range(num_iter):

        for ep in range(num_eps):

            # budget = np.copy(env_config['init_budget']())
            cur_dict = traj[traj_index]
            budget = cur_dict['oldState'][:num_commodities].copy()
            X_alg = np.zeros((num_steps, num_types, num_commodities))
            sizes = np.zeros((num_steps, num_types))

            for step in range(num_steps):
                cur_dict = traj[traj_index]

                X_alg[step] = cur_dict['action'].copy()
                sizes[step] = cur_dict['oldState'][num_commodities:].copy()
                traj_index += 1

            max_envy = 0

            for theta1 in range(num_types):
                for t1 in range(num_steps):
                    for theta2 in range(num_types):
                        for t2 in range(num_steps):
                            max_envy = max(max_envy, np.abs(utility_function(
                                X_alg[t2, theta2], weight_matrix[theta2]) - utility_function(X_alg[t1, theta1], weight_matrix[theta1])))

            final_avg_envy[ep] += max_envy

    return (-1)*np.mean(final_avg_envy)


def delta_COUNTERFACTUAL_ENVY(traj, env_config):
    weight_matrix = env_config['weight_matrix']
    utility_function = env_config['utility_function']
    num_iter = traj[-1]['iter']+1
    num_eps = traj[-1]['episode']+1
    num_steps = traj[-1]['step']+1
    num_types, num_commodities = traj[-1]['action'].shape
    final_avg_envy = np.zeros(num_eps)

    prob, solver = generate_cvxpy_solve(num_types, num_commodities)
    traj_index = 0
    for iter_num in range(num_iter):

        for ep in range(num_eps):

            # budget = np.copy(env_config['init_budget']())
            cur_dict = traj[traj_index]
            budget = cur_dict['oldState'][:num_commodities].copy()
            X_alg = np.zeros((num_steps, num_types, num_commodities))
            sizes = np.zeros((num_steps, num_types))

            for step in range(num_steps):
                cur_dict = traj[traj_index]

                X_alg[step] = cur_dict['action'].copy()
                sizes[step] = cur_dict['oldState'][num_commodities:].copy()
                traj_index += 1

            X_opt = offline_opt(budget, sizes, weight_matrix, solver)

            max_envy = 0

            for theta in range(num_types):
                for t in range(num_steps):
                    max_envy = max(max_envy, np.abs(utility_function(
                        X_opt[t, theta], weight_matrix[theta]) - utility_function(X_alg[t, theta], weight_matrix[theta])))

            final_avg_envy[ep] += max_envy

    return (-1)*np.mean(final_avg_envy)


def delta_EXANTE_ENVY(traj, env_config):
    weight_matrix = env_config['weight_matrix']
    utility_function = env_config['utility_function']
    num_iter = traj[-1]['iter']+1
    num_eps = traj[-1]['episode']+1
    num_steps = traj[-1]['step']+1
    num_types, num_commodities = traj[-1]['action'].shape
    final_avg_envy = np.zeros(num_eps)

    prob, solver = generate_cvxpy_solve(num_types, num_commodities)

    X_alg = np.zeros((num_iter, num_eps, num_steps,
                     num_types, num_commodities))
    X_opt = np.zeros((num_iter, num_eps, num_steps,
                     num_types, num_commodities))

    traj_index = 0
    for iter_num in range(num_iter):

        for ep in range(num_eps):

            # budget = np.copy(env_config['init_budget']())
            cur_dict = traj[traj_index]
            budget = cur_dict['oldState'][:num_commodities].copy()
            sizes = np.zeros((num_steps, num_types))

            for step in range(num_steps):
                cur_dict = traj[traj_index]

                X_alg[iter_num, ep, step] = cur_dict['action'].copy()
                sizes[step] = cur_dict['oldState'][num_commodities:].copy()
                traj_index += 1

            X_opt[iter_num, ep] = offline_opt(
                budget, sizes, weight_matrix, solver)

    for ep in range(num_eps):
        max_envy = 0
        for theta in range(num_types):
            for t in range(num_steps):
                avg_diff = 0
                for iter_num in range(num_iter):
                    avg_diff += utility_function(X_opt[iter_num, ep, theta], weight_matrix[theta]) - utility_function(
                        X_alg[iter_num, ep, t, theta], weight_matrix[theta])
                max_envy = max(max_envy, (1/num_iter)*avg_diff)
        final_avg_envy[ep] = max_envy
    return (-1)*np.mean(final_avg_envy)
