import numpy as np

def random_risk(contact_network, fraction_infected = 0.01, ensemble_size=1):

    population = len(contact_network)
    states_ensemble = np.zeros([ensemble_size, 5 * population])
    for mm in range(ensemble_size):
        infected = np.random.choice(population, replace = False, size = int(population * fraction_infected))
        E, I, H, R, D = np.zeros([5, population])
        S = np.ones(population,)
        I[infected] = 1.
        S[infected] = 0.

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

    return states_ensemble

def uniform_risk(contact_network, fraction_infected = 0.01, ensemble_size=1):

    population = len(contact_network)
    states_ensemble = np.zeros([ensemble_size, 5 * population])
    for mm in range(ensemble_size):
        S, E, I, H, R, D = np.zeros([6, population])
        I += fraction_infected
        S += 1 - fraction_infected

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

    return states_ensemble

def deterministic_risk(contact_network, initial_states, ensemble_size=1):

    population = len(contact_network)
    states_ensemble = np.zeros([ensemble_size, 5 * population])

    init_catalog = {'S': False, 'I': True}
    infected = np.array([init_catalog[status] for status in list(initial_states.values())])

    for mm in range(ensemble_size):
        E, I, H, R, D = np.zeros([5, population])
        S = np.ones(population,)
        I[infected] = 1.
        S[infected] = 0.

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

    return states_ensemble
