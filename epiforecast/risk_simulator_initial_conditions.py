import numpy as np

from .utilities import mask_by_compartment

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

def uniform_risk(population, fraction_infected = 0.01, ensemble_size=1):
    states_ensemble = np.zeros([ensemble_size, 5 * population])
    for mm in range(ensemble_size):
        S, E, I, H, R, D = np.zeros([6, population])
        I += fraction_infected
        S += 1 - fraction_infected

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

    return states_ensemble, 'urisk'+str(int(fraction_infected*100)).zfill(3)

def deterministic_risk(
        population,
        initial_states,
        ensemble_size=1):
    """
    Generate ICs suitable for master equation ensemble from kinetic model states

    Input:
        population (int): total number of nodes in a network
        initial_states (dict): a mapping node_number -> state
        ensemble_size (int): size of the ensemble

    Output:
        ensemble_state (np.array): (ensemble_size, 5*population) array of
                                    states, each ensemble member replicates
                                    initial_states
    """
    # TODO population = len(initial_states), no need to pass it
    susceptible_mask  = mask_by_compartment(initial_states, 'S')
    infected_mask     = mask_by_compartment(initial_states, 'I')
    hospitalized_mask = mask_by_compartment(initial_states, 'H')
    resistant_mask    = mask_by_compartment(initial_states, 'R')
    dead_mask         = mask_by_compartment(initial_states, 'D')

    S, I, H, R, D = np.zeros([5, population])

    S[susceptible_mask]  = 1.
    I[infected_mask]     = 1.
    H[hospitalized_mask] = 1.
    R[resistant_mask]    = 1.
    D[dead_mask]         = 1.

    member_state = np.hstack((S, I, H, R, D))
    ensemble_state = np.tile(member_state, (ensemble_size, 1))

    return ensemble_state

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


