import numpy as np
import cvxpy as cp
import pandas as pd
import or_suite


'''

Helper code to run a single simulation of either an ORSuite experiment or the wrapper for a stable baselines algorithm.

'''

def run_single_algo(env, agent, settings):
    '''
    Runs a single experiment

    Inputs:
        env - environment
        agent - agent
        setting - dictionary containing experiment settings
    '''
    exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()

def run_single_algo_tune(env, agent, scaling_list, settings):
    best_reward = (-1)*np.inf
    best_scaling = scaling_list[0]

    for scaling in scaling_list:
        agent.reset()
        agent.scaling = scaling

        exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
        exp.run()
        dt = pd.DataFrame(exp.data, columns=['episode', 'iteration', 'epReward', 'memory', 'time'])
        avg_end_reward = dt[dt['episode'] == dt.max()['episode']].iloc[0]['epReward']
        if avg_end_reward >= best_reward:
            best_reward = avg_end_reward
            best_scaling = scaling_list[0]
            best_exp = exp
    best_exp.save_data()
    print(best_scaling)

# Helper code to run single stable baseline experiment

def run_single_sb_algo(env, agent, settings):
    '''
    Runs a single experiment

    Inputs:
        env - environment
        agent - agent
        setting - dictionary containing experiment settings
    '''


    exp = or_suite.experiment.sb_experiment.SB_Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()



'''
PROBLEM DEPENDENT METRICS

Sample implementation of problem dependent metrics.  Each one of them should take in a trajectory (as output and saved in an experiment)
and return a corresponding value, where large corresponds to 'good'.

'''


'''

AMBULANCE ENVIRONMENT

'''

# Calculating mean response time for ambulance environment on the trajectory datafile
def mean_response_time(traj, dist):
    mrt = 0
    for i in range(len(traj)):
        cur_data = traj[i]
        mrt += (-1)*np.min(dist(np.array(cur_data['action']),cur_data['info']['arrival']))
    return mrt / len(traj)

# Calculating the variance in the response time for ambulance environment on the trajectory datafile
def response_time_variance(traj, dist):
    dists = []
    for i in range(len(traj)):
        cur_data = traj[i]
        dists.append(np.min(dist(np.array(cur_data['action']),cur_data['info']['arrival'])))
    return (-1)*np.var(dists)




'''

RESOURCE ALLOCATION ENVIRONMENT

'''


def offline_opt(budget, size, weights, solver):
    """
    Uses solver from generate_cvxpy_solve and applies it to values
    
    Inputs:
        budget: initial budget for K commodities
        size: 2D numpy array of sizes of each type at each location
        weights: 2D numpy array containing the demands of each type
    """
    tot_size = np.sum(size, axis=0)
    _, x = solver(tot_size, weights, budget)
    allocation = np.zeros((size.shape[0], weights.shape[0], weights.shape[1]))
    for i in range(size.shape[0]):
        allocation[i,:,:] = x
    return allocation


def generate_cvxpy_solve(num_types, num_resources):
    """
    Creates a generic solver to solve the offline resource allocation problem
    
    Inputs: 
        num_types - number of types
        num_resources - number of resources
    Returns:
        prob - CVXPY problem object
        solver - function that solves the problem given data
    """
    x = cp.Variable(shape=(num_types,num_resources))
    sizes = cp.Parameter(num_types, nonneg=True)
    weights = cp.Parameter((num_types, num_resources), nonneg=True)
    budget = cp.Parameter(num_resources, nonneg=True)
    objective = cp.Maximize(cp.log(cp.sum(cp.multiply(x, weights), axis=1)) @ sizes)
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
            # print(ep)
            budget = np.copy(env_config['init_budget'])
            # print('budget:' + str(budget))
            for step in range(num_steps):
                cur_dict = traj[traj_index]
                # print(cur_dict['oldState'])
                # print(cur_dict['action'])
                budget -= np.matmul(cur_dict['oldState'][num_commodities:], cur_dict['action'])
                traj_index += 1
#                budget -= cur_dict['oldState']
            final_avg_efficiency[ep] += np.sum(budget)
    return (-1)*np.mean(final_avg_efficiency)


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

            budget = np.copy(env_config['init_budget'])
            X_alg = np.zeros((num_steps, num_types, num_commodities))
            sizes = np.zeros((num_steps, num_types))

            for step in range(num_steps):
                cur_dict = traj[traj_index]

                X_alg[step] = cur_dict['action']
                sizes[step] = cur_dict['oldState'][num_commodities:]
                traj_index += 1

            
            prop_alloc = budget / np.sum(sizes)
            max_prop = 0

            for theta in range(num_types):
                for h in range(num_steps):
                    max_prop = max(max_prop, utility_function(prop_alloc, weight_matrix[theta, :]) - utility_function(X_alg[h, theta], weight_matrix[theta,:]))


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

            budget = np.copy(env_config['init_budget'])
            X_alg = np.zeros((num_steps, num_types, num_commodities))
            sizes = np.zeros((num_steps, num_types))

            for step in range(num_steps):
                cur_dict = traj[traj_index]

                X_alg[step] = cur_dict['action']
                sizes[step] = cur_dict['oldState'][num_commodities:]
                traj_index += 1

            
            max_envy = 0

            for theta1 in range(num_types):
                for t1 in range(num_steps):
                    for theta2 in range(num_types):
                        for t2 in range(num_steps):
                            max_envy = max(max_envy, utility_function(X_alg[t2, theta2], weight_matrix[theta1]) - utility_function(X_alg[t1, theta1], weight_matrix[theta1]))

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

            budget = np.copy(env_config['init_budget'])
            X_alg = np.zeros((num_steps, num_types, num_commodities))
            sizes = np.zeros((num_steps, num_types))

            for step in range(num_steps):
                cur_dict = traj[traj_index]

                X_alg[step] = cur_dict['action']
                sizes[step] = cur_dict['oldState'][num_commodities:]
                traj_index += 1

            
            X_opt = offline_opt(budget, sizes, weight_matrix, solver)

            max_envy = 0

            for theta in range(num_types):
                for t in range(num_steps):
                    max_envy = max(max_envy, np.abs(utility_function(X_opt[t, theta], weight_matrix[theta]) - utility_function(X_alg[t, theta], weight_matrix[theta])))

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

    X_alg = np.zeros((num_iter, num_eps, num_steps, num_types, num_commodities))
    X_opt = np.zeros((num_iter, num_eps, num_steps, num_types, num_commodities))
    
    traj_index = 0
    for iter_num in range(num_iter):

        for ep in range(num_eps):

            budget = np.copy(env_config['init_budget'])
            sizes = np.zeros((num_steps, num_types))

            for step in range(num_steps):
                cur_dict = traj[traj_index]

                X_alg[iter_num, ep, step] = cur_dict['action']
                sizes[step] = cur_dict['oldState'][num_commodities:]
                traj_index += 1

            
            X_opt[iter_num, ep] = offline_opt(budget, sizes, weight_matrix, solver)


    for ep in range(num_eps):
        max_envy = 0
        for theta in range(num_types):
            for t in range(num_steps):
                avg_diff = 0
                for iter_num in range(num_iter):
                    avg_diff += utility_function(X_opt[iter_num, ep, theta], weight_matrix[theta]) - utility_function(X_alg[iter_num, ep, t, theta], weight_matrix[theta])
                max_envy = max(max_envy, (1/num_iter)*avg_diff)
        final_avg_envy[ep] = max_envy
    return (-1)*np.mean(final_avg_envy)