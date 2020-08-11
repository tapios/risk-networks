import numpy as np

from .utilities import mask_by_compartment, dict_slice, shuffle

def random_risk(population, fraction_infected = 0.01, ensemble_size=1):
    states_ensemble = np.zeros([ensemble_size, 5 * population])
    for mm in range(ensemble_size):
        infected = np.random.choice(population, replace = False, size = int(population * fraction_infected))
        E, I, H, R, D = np.zeros([5, population])
        S = np.ones(population,)
        I[infected] = 1.
        S[infected] = 0.

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

    return states_ensemble, 'rrisk'+str(int(fraction_infected*100)).zfill(3)

def random_risk_range(population, min_fraction_infected = None, max_fraction_infected=None, ensemble_size=1):
    """
    Initial ensemble states have a fraction of infected, given by a uniform distn between a provided min and max
    """
    if min_fraction_infected is None:
        min_fraction_infected = 1.001/population #to avoid rounding errors
    if max_fraction_infected is None:
        max_fraction_infected = min_fraction_infected

    assert (max_fraction_infected >= min_fraction_infected) 

    fraction_infected = np.random.uniform(min_fraction_infected, max_fraction_infected, ensemble_size)

    states_ensemble = np.zeros([ensemble_size, 5 * population])
    for mm in range(ensemble_size):
        infected = np.random.choice(population, replace = False, size = int(population * fraction_infected[mm]))
        E, I, H, R, D = np.zeros([5, population])
        S = np.ones(population,)
        I[infected] = 1.
        S[infected] = 0.

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

    return states_ensemble



def uniform_risk(population, fraction_infected = 0.01, ensemble_size=1):
    states_ensemble = np.zeros([ensemble_size, 5 * population])
    for mm in range(ensemble_size):
        S, E, I, H, R, D = np.zeros([6, population])
        I += fraction_infected
        S += 1 - fraction_infected

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

    return states_ensemble, 'urisk'+str(int(fraction_infected*100)).zfill(3)

def deterministic_risk(
        user_nodes,
        kinetic_states,
        ensemble_size):
    """
    Generate exact ensemble user states from full kinetic model states

    The resulting ensemble user states are exact copies of the kinetic model
    states

    Input:
        user_nodes (np.array): (n_user_nodes,) array of user node indices
        kinetic_states (dict): a mapping node -> state
        ensemble_size (int): size of the ensemble

    Output:
        ensemble_state (np.array): (ensemble_size, 5*n_user_nodes) array of
                                   states
    """
    user_kinetic_states = dict_slice(kinetic_states, user_nodes)
    member_state = __kinetic_to_master(user_kinetic_states)
    ensemble_state = np.tile(member_state, (ensemble_size, 1))

    return ensemble_state

def kinetic_to_master_same_fraction(
        user_nodes,
        kinetic_states,
        ensemble_size):
    """
    Generate ensemble user states from full kinetic model states (fraction)

    The resulting ensemble states have the same fraction of nodes in each
    compartment as the kinetic model states, but not exactly the same nodes

    Input:
        user_nodes (np.array): (n_user_nodes,) array of user node indices
        kinetic_states (dict): a mapping node -> state
        ensemble_size (int): size of the ensemble

    Output:
        ensemble_state (np.array): (ensemble_size, 5*n_user_nodes) array of
                                   states
    """
    user_kinetic_states = dict_slice(kinetic_states, user_nodes)

    n_user_nodes = len(user_kinetic_states)
    ensemble_state = np.empty( (ensemble_size, 5*n_user_nodes) )

    for m in range(ensemble_size):
        shuffled_user_kinetic_states = shuffle(user_kinetic_states)
        ensemble_state[m] = __kinetic_to_master(shuffled_user_kinetic_states)

    return ensemble_state

def __kinetic_to_master(kinetic_states):
    """
    Generate a vector of master states from kinetic model states

    The resulting vector has {0, 1} values which correspond to kinetic states

    Input:
        kinetic_states (dict): a mapping node -> state

    Output:
        master_states (np.array): (5*n_nodes,) array of states
    """
    s_mask = mask_by_compartment(kinetic_states, 'S')
    i_mask = mask_by_compartment(kinetic_states, 'I')
    h_mask = mask_by_compartment(kinetic_states, 'H')
    r_mask = mask_by_compartment(kinetic_states, 'R')
    d_mask = mask_by_compartment(kinetic_states, 'D')

    n_nodes = len(kinetic_states)
    S, I, H, R, D = np.zeros([5, n_nodes])

    S[s_mask] = 1.
    I[i_mask] = 1.
    H[h_mask] = 1.
    R[r_mask] = 1.
    D[d_mask] = 1.

    master_states = np.hstack((S, I, H, R, D))

    return master_states

def prevalence_deterministic_risk(population, initial_states, ensemble_size=1):
    states_ensemble = np.zeros([ensemble_size, 5 * population])

    init_catalog = {'S': False, 'I': True}
    infected = [init_catalog[status] for status in list(initial_states.values())]
    fraction_infected = np.array(infected).sum()/population

    for mm in range(ensemble_size):
        E, I, H, R, D = np.zeros([5, population])
        S = np.ones(population,)
        I[infected] = 0.44
        S[infected] = 1 - 0.44

        I[not infected] = 0.002
        S[not infected] = 1 - 0.002

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

        return states_ensemble, 'pdrisk'+str(int(fraction_infected*100)).zfill(3)

def prevalence_random_risk(population, fraction_infected = 0.01, ensemble_size=1):
    states_ensemble = np.zeros([ensemble_size, 5 * population])
    for mm in range(ensemble_size):
        infected = np.random.choice(population, replace = False, size = int(population * fraction_infected))
        infected = [True if node in infected else False for node in range(population)]
        E, I, H, R, D = np.zeros([5, population])
        S = np.ones(population,)
        I[infected] = 0.44
        S[infected] = 1 - 0.44

        I[not infected] = 0.002
        S[not infected] = 1 - 0.002

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

    return states_ensemble, 'prrisk'+str(int(fraction_infected*100)).zfill(3)


