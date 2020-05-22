import numpy as np
import contextlib

n_states = 5

# Index guide:
#
# 0: Susceptible
# 1: Exposed
# 2: Infected
# 3: Hospitalized
# 4: Resistant
# (5: Decesased)
susceptible  = s = 0
infected     = i = 1
hospitalized = h = 2
resistant    = r = 3
deceased     = d = 4

@contextlib.contextmanager
def temporary_seed(seed):
    """
    Temporarily changes the global random state, and then changes it back.

    Ref: https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    """

    state = np.random.get_state()

    np.random.seed(seed)

    try:
        yield
    finally:
        np.random.set_state(state)


def random_infection(population, infected=10):
    """
    Returns an `np.array` corresponding to the epidemiological state of a population.

    Each person can be in 1 of 5 states, so `state.shape = (5, population)`.
    """

    # The states are S, E, I, H, R (, D)
    state = np.zeros((n_states, population))

    # Some are infected...
    infected = np.random.choice(population, infected)
    state[i, infected] = 1

    # ... and everyone else is susceptible
    state[s, :] = 1 - state[i, :]

    return state

def midnight_on_Tuesday(kinetic_model, 
                            percent_infected = 0.1,
                             percent_exposed = 0.05,
                                 random_seed = 1234569,
                        ):
    """
    Returns an `np.array` corresponding to the epidemiological state of a population
    "at midnight on Tuesday".

    Each person can be in 1 of 5 states, so `state.shape = (6, population)`.
    """

    population = kinetic_model.population

    n_infected = int(np.round(percent_infected * population))
    n_exposed = int(np.round(percent_exposed * population))

    # Generate random indices for infected, and exposed
    with temporary_seed(random_seed):
        infected = np.random.choice(population, n_infected)
        exposed = np.random.choice(population, n_exposed)

    state = np.zeros((n_states, population))

    # Some are infected...
    state[i, infected] = 1 
    state[i, exposed] = 0 # (except those who are exposed)

    # and everyone else is susceptible.
    state[s, :] = 1 - state[i, :]

    # (except those who are exposed).
    state[s, exposed] = 0

    # (We may want to identify a hospitalized group as well.)

    return state

def state_distribution_at_midnight_on_Tuesday():
    pass

def transition_rates_distribution_at_midnight_on_Tuesday():
    pass

def transmission_rates_distribution_at_midnight_on_Tuesday():
    pass
