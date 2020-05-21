import numpy as np

def random_infection(population, infected=10):
    """
    Returns an `np.array` corresponding to the epidemiological state of a population.

    Each person can be in 1 of 6 states, so `state.shape = (6, population)`.
    """

    state = np.zeros((6, population))

    # The states are S, E, I, H, R, D
    state[1, :] = 1 # everyone is susceptible at first...

    # ...except for those who are infected.
    infected = np.random.choice(population, infected)
    state[3, infected] = 1
    state[1, infected] = 0

    return state

def midnight_on_Tuesday(kinetic_model):
    pass

def state_distribution_at_midnight_on_Tuesday():
    pass

def transition_rates_distribution_at_midnight_on_Tuesday():
    pass

def transmission_rates_distribution_at_midnight_on_Tuesday():
    pass
