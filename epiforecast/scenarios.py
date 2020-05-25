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

# Our state is a 1D vector. Thus, accessing the values for a particular state requires
# slicing into this vector. These functions return the appropritate subranges for each state.
def susecptible_indices(population):  return np.arange(start = 0 * population, stop = 1 * population)
def infected_indices(population):     return np.arange(start = 1 * population, stop = 2 * population)
def hospitalized_indices(population): return np.arange(start = 2 * population, stop = 3 * population)
def resistant_indices(population):    return np.arange(start = 3 * population, stop = 4 * population)
def deceased_indices(population):     return np.arange(start = 4 * population, stop = 5 * population)

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

    Each person can be in 1 of 5 states, so `state.shape = (5 * population)`.
    """

    # The states are S, E, I, H, R (, D)
    state = np.zeros((n_states * population,))
    i = infected_indices(population)
    s = infected_indices(population)

    # Some are infected...
    infected = np.random.choice(population, infected)
    i_infected = i[infected]
    state[i_infected] = 1

    # ... and everyone else is susceptible
    state[s] = 1 - state[i]

    return state

def midnight_on_Tuesday(kinetic_model, 
                            percent_infected = 0.1,
                             percent_exposed = 0.05,
                                 random_seed = 1234569,
                        ):
    """
    Returns an `np.array` corresponding to the epidemiological state of a population
    "at midnight on Tuesday".

    Each person can be in 1 of 5 states, so `state.shape = (5, population)`.
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

def percent_infected_at_midnight_on_Tuesday():
    return 0.01

def transition_rates_distribution_at_midnight_on_Tuesday():
    pass

def transmission_rates_distribution_at_midnight_on_Tuesday():
    pass

def randomly_infected_ensemble(n_ensemble, population, percent_infected, random_seed=1234):
    """
    Returns an ensemble of states of 5 x population. In each member of the ensemble,
    `percent_infected` of the population is infected.

    Args
    ----

    n_ensemble: The number of ensemble members.

    population: The population. Each ensemble has a state of size `5 * population`. The total
                size of the ensemble of states is `n_ensemble x 5 * population`.

    percent_infected: The percent of people who are infected in each ensemble member.                    

    random_seed: A random seed so that the outcome is reproducible.
    """

    n_infected = int(np.round(population * percent_infected))

    # Extract the indices corresponding to infected and susceptible states
    s = susceptible_indices(population)
    i = infected_indices(population) # is needed later

    # Initialize states with susceptible = 1.
    states = np.zeros([n_ensemble, n_states * population])
    states[:, s] = 1

    with temporary_seed(random_seed):
        for m in range(n_ensemble):

            # Select random indices from a list of indices = [0, population)
            randomly_infected = np.random.choice(population, size=n_infected)

            # Translate random indices to indices of the infected state
            i_randomly_infected = i[randomly_infected]
            s_randomly_infected = s[randomly_infected]

            # Infect the people
            states[m, i_randomly_infected] = 1
            states[m, s_randomly_infected] = 0

    return states
