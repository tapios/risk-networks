import numpy as np
import contextlib

n_states = 6

# Index guide:
#
# 0: Susceptible
# 1: Exposed
# 2: Infected
# 3: Hospitalized
# 4: Resistant
# (5: Decesased)
Susceptible  = S = 0
Exposed      = E = 1
Infected     = I = 2
Hospitalized = H = 3
Resistant    = R = 4
Deceased     = D = 5

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

    Each person can be in 1 of 6 states, so `state.shape = (6, population)`.
    """

    # The states are S, E, I, H, R (, D)
    state = np.zeros((n_states, population))

    # Some are infected...
    infected = np.random.choice(population, infected)
    state[I, infected] = 1

    # ... and everyone else is susceptible
    state[S, :] = 1 - state[I, :]

    return state

def midnight_on_Tuesday(kinetic_model, 
                            percent_infected = 0.1,
                             percent_exposed = 0.05,
                                 random_seed = 1234569,
                        ):
    """
    Returns an `np.array` corresponding to the epidemiological state of a population
    "at midnight on Tuesday".

    Each person can be in 1 of 6 states, so `state.shape = (6, population)`.
    """

    population = kinetic_model.population

    n_infected = int(np.round(percent_infected * population))
    n_exposed = int(np.round(percent_exposed * population))

    # Generate random indices for infected, and exposed
    with temporary_seed(random_seed):
        infected = np.random.choice(population, n_infected)
        exposed = np.random.choice(population, n_exposed)

    print(infected)
    print(exposed)

    state = np.zeros((n_states, population))

    # Some are exposed ...
    state[E, exposed] = 1

    # ... some are infected...
    state[I, infected] = 1 
    state[I, exposed] = 0 # (except those who are exposed)

    # ... and everyone else is susceptible.
    state[S, :] = 1 - state[E, :] - state[I, :]

    # (We may want to identify a hospitalized group as well.)

    return state

def state_distribution_at_midnight_on_Tuesday():
    pass

def transition_rates_distribution_at_midnight_on_Tuesday():
    pass

def transmission_rates_distribution_at_midnight_on_Tuesday():
    pass
