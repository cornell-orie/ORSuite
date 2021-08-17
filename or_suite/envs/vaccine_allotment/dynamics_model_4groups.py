"""
Adapted from code by Cornell University students Mohammad Kamil (mk848), Carrie Rucker (cmr284), Jacob Shusko (jws383), Kevin Van Vorst (kpv23).
"""
import numpy as np
#import random
master_seed = 1


def dynamics_model(params, population):
    """
    A function to run SIR disease dynamics for 4 groups.

    Args:
        params: (dict) a dictionary containing the following keys and values:

            - contact_matrix: (np.array of floats) Contact rates between susceptible people in each class and the infected people.

            - P: (np.array of floats) P = [p1 p2 p3 p4] where pi = Prob(symptomatic | infected) for a person in class i.

            - H: (np.array of floats) H = [h1 h2 h3 h4] where hi = Prob(hospitalized | symptomatic) for a person in class i.

            - beta: (float) Recovery rate.

            - gamma: (int) Vaccination rate.

            - vaccines: The (int) number of vaccine available for this time period.

            - priority: The (list of chars) vaccination priority order of the four groups.

            - time_step: The (float) number of units of time you want the simulation to run for e.g. if all your rates are per day and you want to simulate 7 days, time_step = 7.

        population : (np.array of ints) The starting state [S1 S2 S3 S4 A1 A2 A3 A4 I H R].

    Returns:
        np.array of ints, dict: 
        newState: the final state [S1 S2 S3 S4 A1 A2 A3 A4 I H N]. Note that instead of returning the final number of recovered people R, we return N, the number of infections that occurred.

        output_dictionary : A (dict) dictionary containing the following keys and values:

            - clock times: List of the times that each event happened.

            - c1 asymptomatic: List of counts of A1 for each time in clks.

            - c2 asymptomatic: List of counts of A2 for each time in clks.

            - c3 asymptomatic: List of counts of A3 for each time in clks.

            - c4 asymptomatic: List of counts of A4 for each time in clks.

            - mild symptomatic: List of counts of I for each time in clks.

            - hospitalized: List of counts of H for each time in clks.

            - c1 susceptible: List of counts of S1 for each time in clks.

            - c2 susceptible: List of counts of S2 for each time in clks.

            - c3 susceptible: List of counts of S3 for each time in clks.

            - c4 susceptible: List of counts of S4 for each time in clks.

            - recovered: List of counts of R for each time in clks.

            - total infected: int - Total number of infected (including those that were already infected).

            - total hospitalized: int - Total number of hospitalized individuals (including those that were already hospitalized).

            - vaccines: int - Total number of vaccines left.

            - event counts: np.array - Contains the counts for the number of times each of the 22 events occurred.


    Typical usage example:
        newState, info = dynamics_model(parameters, population)
    """

    # extract arguments from params dictionary
    state = np.copy(population)
    P = params['P']
    H = params['H']
    LAMBDA = params['contact_matrix']
    gamma = params['gamma']
    beta = params['beta']
    priority = params['priority']
    vaccines = params['vaccines']
    time_step = params['time_step']

    # output tracking
    clks = [0]
    c1_Ss = [state[0]]
    c2_Ss = [state[1]]
    c3_Ss = [state[2]]
    c4_Ss = [state[3]]
    c1_infs = [state[4]]
    c2_infs = [state[5]]
    c3_infs = [state[6]]
    c4_infs = [state[7]]
    Is_infs = [state[8]]
    Hs_infs = [state[9]]
    Rs = [state[10]]

    # first priority group
    # if priority list is empty, the policy is random vaccination
    if len(priority) != 0:
        priority_group = int(priority[0]) - 1
        priority.pop(0)
        randomFlag = False
    else:
        # to begin, assume all groups are eligible; lose eligbility if no susceptible people left in that group
        eligible_list = [0, 1, 2, 3]
        eligible_array = np.array(eligible_list)
        priority_group = np.random.choice(eligible_array)
        randomFlag = True

    # possible state changes
    # each key correponds to the index of an event in rates and has a value [i,j]
    # the state change is state[i]-- and state[j]++
    state_changes = {0: [0, 4], 1: [0, 8], 2: [0, 9], 3: [1, 5],
                     4: [1, 8], 5: [1, 9], 6: [2, 6], 7: [2, 8],
                     8: [2, 9], 9: [3, 7], 10: [3, 8], 11: [3, 9],
                     12: [4, 10], 13: [5, 10], 14: [6, 10], 15: [7, 10],
                     16: [8, 10], 17: [9, 10], 18: [0, 10], 19: [1, 10],
                     20: [2, 10], 21: [3, 10]}

    # rates for all 22 events
    rates = np.zeros(shape=(22,))

    # counts for each of the 22 events
    event_counts = np.zeros(shape=(22,))

    # compute the probabilities associated with each of the 12 infection rates
    probs = np.zeros(shape=(12,))
    probs[[0, 3, 6, 9]] = 1 - P
    probs[[1, 4, 7, 10]] = np.multiply(P, 1-H)
    probs[[2, 5, 8, 11]] = np.multiply(P, H)

    # compute the rates for the 12 infection events
    temp = np.matmul(np.diag(state[0:4]), LAMBDA)
    inf_rates = np.matmul(temp, state[4:10])
    inf_rates = np.repeat(inf_rates, repeats=3, axis=0)
    rates[0:12] = np.multiply(probs, inf_rates)

    # compute the rates for the 6 recovery events
    rates[12:18] = beta * state[4:10]

    # compute the rates for the vaccination events
    rates[priority_group + 18] = gamma

    # flag - if true, we have not run out of vaccines or people to vaccinate yet
    #      - if false, either there are no vaccines left or no people to vaccinate
    # once set to False, it remains False
    vaccFlag = True

    rate_sum = np.sum(rates)

    # exponential timer
    nxt = np.random.exponential(1/rate_sum)
    clk = 0

    # maximum number of vaccination events that we want to happen
    max_vacc_events = gamma*time_step

    # We will simulate the Markov chain until we've reached max_vacc_events vaccination events
    while np.sum(event_counts[18:22]) < max_vacc_events:
        clk += nxt

        # get the index of the event that is happening
        index = np.random.choice(22, 1, p=rates/rate_sum)[0]

        # if this is a vaccination event, call vacc_update
        # otherwise, simple state change
        if index in np.arange(18, 22):
            if randomFlag:
                state, event_counts, priority_group, eligible_list, vaccines = rand_vacc_update(state=state,
                                                                                                changes=state_changes,
                                                                                                group=priority_group,
                                                                                                eligible=eligible_list,
                                                                                                vaccines=vaccines,
                                                                                                count=event_counts)

            else:
                state, event_counts, vaccFlag, priority_group, priority, vaccines = vacc_update(state=state,
                                                                                                changes=state_changes,
                                                                                                ind=index,
                                                                                                count=event_counts,
                                                                                                flag=vaccFlag,
                                                                                                group=priority_group,
                                                                                                priority=priority,
                                                                                                vaccines=vaccines)
            # update vaccination rate
            rates[18:22] = np.zeros(shape=(4,))
            rates[priority_group+18] = gamma

            # for testing
            if (np.sum(event_counts[18:22]) % gamma) == 0:
                print(" We've reached vaccination event number " +
                      str(np.sum(event_counts[18:22])))
        else:
            state[state_changes[index][0]] -= 1
            state[state_changes[index][1]] += 1
            event_counts[index] += 1

        # for debugging: we should never have negative numbers in a state!
        assert np.all(state >= 0), 'Accepted negative state change'

        # update infection and recovery rates
        # 12 infection events
        temp = np.matmul(np.diag(state[0:4]), LAMBDA)
        inf_rates = np.matmul(temp, state[4:10])
        inf_rates = np.repeat(inf_rates, repeats=3, axis=0)
        rates[0:12] = np.multiply(probs, inf_rates)

        # 6 recovery events
        rates[12:18] = beta * state[4:10]

        rate_sum = np.sum(rates)

        # TODO: not sure if this conditional is necessary
        if rate_sum > 0:
            nxt = np.random.exponential(1/rate_sum)
        else:
            print("The sum of the rates is less than or equal to zero!")
            break

        # output tracking
        clks.append(clk)
        c1_Ss.append(state[0])
        c2_Ss.append(state[1])
        c3_Ss.append(state[2])
        c4_Ss.append(state[3])
        c1_infs.append(state[4])
        c2_infs.append(state[5])
        c3_infs.append(state[6])
        c4_infs.append(state[7])
        Is_infs.append(state[8])
        Hs_infs.append(state[9])
        Rs.append(state[10])

        # if there are no more infected individuals, the simulation should end
        if np.sum(state[4:11]) <= 0:
            print("Reached a disease-free state on day " + str(clk))

    new_infections = np.sum(event_counts[0:12])
    total_infected = new_infections + np.sum(population[4:10])
    total_hospitalized = np.sum(event_counts[[2, 5, 8, 11]]) + population[9]
    total_recovered = state[10]
    newState = state
    newState[10] = new_infections  # return new infections instead of recovered

    output_dictionary = {'clock times': clks, 'c1 asymptomatic': c1_infs, 'c2 asymptomatic': c2_infs, 'c3 asymptomatic': c3_infs,
                         'c4 asymptomatic': c4_infs, 'mild symptomatic': Is_infs, 'hospitalized': Hs_infs, 'c1 susceptible': c1_Ss,
                         'c2 susceptible': c2_Ss, 'c3 susceptible': c3_Ss, 'c4 susceptible': c4_Ss, 'recovered': Rs, 'total infected': total_infected,
                         'total hospitalized': total_hospitalized, 'total recovered': total_recovered, 'vaccines': vaccines, 'event counts': event_counts}
    return newState, output_dictionary


def vacc_update(state, changes, ind, count, flag, group, priority, vaccines):
    """
    Performs a vaccination according to the priority order and updates the environment accordingly.

    Args:
        state : np.array -
            The state of the environment when the function is called.
        changes : dict -
            Possible state changes [i,j] where the change is state[i]--, state[j]++
        ind : int -
            The index corresponding the current vaccination event.
        count : np.array -
            Counts of all the events.
        flag : boolean -
            If true, we have vaccines and people to vaccinate
            if false, either there are no vaccines left or no people to vaccinate.
        group : int -
            Current priority group; always either 0, 1, 2 or 3.
        priority : List of strings,
            priority order list.
        vaccines : int -
            Vaccine count.

    Returns:
        np.array, int, boolean, int, list of strings, int:
        newState : Updated state.

        count : Updated event counts.

        flag : Updated flag for if we should continue to vaccinate.

        group : Updated priority group.

        priority : Updated priority list.

        vaccines : Updated vaccine count.


    Notes:
        This function should only ever be called from inside dynamics model. 
        The vaccination events correspond to indices 18, 19, 20 and 21 in count and changes.
        We increment the event counter even if we technically did not vaccinate anyone. 
    """
    newState = np.copy(state)

    # if flag, up until now there have been vaccines and people to vaccinate
    # otherwise, increment event counter and exit function
    if flag:
        # if there are still vaccines, proceeed with vaccination.
        # otherwise, set flag to false and increment event counter
        if vaccines > 0:
            newState[changes[ind][0]] -= 1
            newState[changes[ind][1]] += 1
            vaccines -= 1

            # while state change was invalid (i.e. we have negative people), undo change & try again
            while np.any(newState < 0):
                newState = np.copy(state)
                vaccines += 1

                # if there are still priority groups left, choose next group & proceed with vaccination
                # otherwise set flag to false; exits for loop
                if len(priority) != 0:
                    group = int(priority[0]) - 1
                    priority.pop(0)
                    newState[changes[group+18][0]] -= 1
                    newState[changes[group+18][1]] += 1
                    vaccines -= 1
                else:
                    flag = False
            # increment event counter
            count[group+18] += 1

        else:
            count[ind] += 1
            flag = False
    else:
        count[ind] += 1
    return newState, count, flag, group, priority, vaccines


def rand_vacc_update(state, changes, group, eligible, vaccines, count):
    """
    Performs a random vaccination and updates the environment accordingly. 

    Args:
        state : np.array -
            The state of the environment when the function is called.
        changes : dict -
            Possible state changes [i,j] where the change is state[i]--, state[j]++
        group : int -
            Current priority group.
        eligible : List of ints,
            groups that are still eligible for vaccination.
        vaccines : int -
            Current number of available vaccines.
        count : np.array -
            Counts of all the events.

    Returns:
        np.array, int, int, list of ints, int:
        state : Updated state.

        count : Updated event count.

        group : Updated priority group.

        eligible : Updated eligible list.

        vaccines : Updated vaccine count.


    Notes:
        This function should only ever be called from inside dynamics model. 
        The vaccination events correspond to indices 18, 19, 20 and 21 in count and changes.
        We increment the event counter even if we technically did not vaccinate anyone. 
    """

    # if there are still people to vaccinate and vaccines left, proceed with vaccination
    # otherwise increment event counter and exit function
    if len(eligible) != 0 and vaccines > 0:
        state[changes[group+18][0]] -= 1
        state[changes[group+18][1]] += 1
        vaccines -= 1

        # while state change was invalid (i.e. we have negative people), undo change & try again
        while np.any(state < 0):
            state[changes[group+18][0]] += 1
            state[changes[group+18][1]] -= 1
            vaccines += 1
            # current group has no one left to vaccinate so remove from eligibilty list
            eligible.remove(group)

            # if there are still eligible groups, choose new priority group and proceed with vaccination
            # otherwise will exit while loop
            if len(eligible) != 0:
                eligible_array = np.array(eligible)
                group = np.random.choice(eligible_array)
                state[changes[group+18][0]] -= 1
                state[changes[group+18][1]] += 1
                vaccines -= 1

        # increment event counter accordingly and choose new priority group if possible
        count[group+18] += 1
        if len(eligible) != 0:
            eligible_array = np.array(eligible)
            group = np.random.choice(eligible_array)
    else:
        count[group+18] += 1
    return state, count, group, eligible, vaccines
