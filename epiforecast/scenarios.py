import numpy as np
import contextlib

from .samplers import AgeAwareBetaSampler, GammaSampler
from .populations import populate_ages, sample_pathological_distribution, TransitionRates

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
def susceptible_indices(population):  return np.arange(start = 0 * population, stop = 1 * population)
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
    s = susceptible_indices(population)

    # Some are infected...
    infected_nodes = np.random.choice(population, infected)
    i_infected = i[infected_nodes]
    state[i_infected] = 1

    # ... and everyone else is susceptible
    state[s] = 1 - state[i]

    return state

def king_county_transition_rates(population, random_seed=1234):
    """
    Returns transition rates for a community of size `population`
    whose statistics vaguely resemble the clinical statistics of 
    King County, WA, USA.
    """

    # ... and the age category of each individual
    age_distribution = [ 0.23,  # 0-19 years
                         0.39,  # 20-44 years
                         0.25,  # 45-64 years
                         0.079  # 65-75 years
                        ]

    # 75 onwards
    age_distribution.append(1 - sum(age_distribution))
    
    with temporary_seed(random_seed):
        ages = populate_ages(population, distribution=age_distribution)

        # Next, we randomly generate clinical properties for our example population.
        # Note that the units of 'periods' are days, and the units of 'rates' are 1/day.
        latent_periods              = sample_pathological_distribution(GammaSampler(k=1.7, theta=2.0), population=population, minimum=2)
        community_infection_periods = sample_pathological_distribution(GammaSampler(k=1.5, theta=2.0), population=population, minimum=1)
        hospital_infection_periods  = sample_pathological_distribution(GammaSampler(k=1.5, theta=3.0), population=population, minimum=1)
        
        hospitalization_fraction     = sample_pathological_distribution(AgeAwareBetaSampler(mean=[ 0.02,  0.17,  0.25, 0.35, 0.45], b=4), ages=ages)
        community_mortality_fraction = sample_pathological_distribution(AgeAwareBetaSampler(mean=[0.001, 0.001, 0.005, 0.02, 0.05], b=4), ages=ages)
        hospital_mortality_fraction  = sample_pathological_distribution(AgeAwareBetaSampler(mean=[0.001, 0.001,  0.01, 0.04,  0.1], b=4), ages=ages)

    transition_rates = TransitionRates(population,
                                       latent_periods,
                                       community_infection_periods,
                                       hospital_infection_periods,
                                       hospitalization_fraction,
                                       community_mortality_fraction,
                                       hospital_mortality_fraction)
    
    return transition_rates

def midnight_on_Tuesday(kinetic_model, 
                            percent_infected = 0.1,
                             percent_exposed = 0.05,
                                 random_seed = 1234,
                        ):
    """
    Returns an `np.array` corresponding to the epidemiological state of a population
    "at midnight on Tuesday".

    Each person can be in 1 of 5 states, so `state.shape = (5, population)`.
    """

    population = kinetic_model.population

    n_infected = int(np.round(percent_infected * population))
    n_exposed = int(np.round(percent_exposed * population))

    # Generate random indices for infected and exposed
    with temporary_seed(random_seed):
        infected_nodes = np.random.choice(population, n_infected)
        exposed_nodes = np.random.choice(population, n_exposed)

    state = np.zeros((n_states * population,))

    i = infected_indices(population)
    s = susceptible_indices(population)

    # Some are infected...
    i_infected = i[infected_nodes]
    state[i_infected] = 1 

    # and everyone else is susceptible.
    state[s] = 1 - state[i]

    # (except those who are exposed).
    s_exposed = s[exposed_nodes]
    state[s_exposed] = 0

    # (We may want to identify a hospitalized group as well.)

    return state

def percent_infected_at_midnight_on_Tuesday():
    return 0.01

def ensemble_transition_rates_at_midnight_on_Tuesday(ensemble_size, population, random_seed=1234):
    transition_rates = []

    for i in range(ensemble_size):
        random_seed += 1
        transition_rates.append(king_county_transition_rates(population, random_seed=random_seed))

    return transition_rates

def ensemble_transmission_rates_at_midnight_on_Tuesday(ensemble_size, random_seed=1234):

    with temporary_seed(random_seed):
        transmission_rates = np.random.uniform(0.04, 0.06, ensemble_size)

    return transmission_rates

def randomly_infected_ensemble(ensemble_size, population, percent_infected, random_seed=1234):
    """
    Returns an ensemble of states of 5 x population. In each member of the ensemble,
    `percent_infected` of the population is infected.

    Args
    ----

    ensemble_size: The number of ensemble members.

    population: The population. Each ensemble has a state of size `5 * population`. The total
                size of the ensemble of states is `ensemble_size x 5 * population`.

    percent_infected: The percent of people who are infected in each ensemble member.                    

    random_seed: A random seed so that the outcome is reproducible.
    """

    n_infected = int(np.round(population * percent_infected))

    # Extract the indices corresponding to infected and susceptible states
    s = susceptible_indices(population)
    i = infected_indices(population) # is needed later

    # Initialize states with susceptible = 1.
    states = np.zeros([ensemble_size, n_states * population])
    states[:, s] = 1

    with temporary_seed(random_seed):
        for m in range(ensemble_size):

            # Select random indices from a list of indices = [0, population)
            randomly_infected = np.random.choice(population, size=n_infected)

            # Translate random indices to indices of the infected state
            i_randomly_infected = i[randomly_infected]
            s_randomly_infected = s[randomly_infected]

            # Infect the people
            states[m, i_randomly_infected] = 1
            states[m, s_randomly_infected] = 0

    return states
