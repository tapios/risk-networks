import os, sys; sys.path.append(os.path.join(".."))

import numpy as np
import networkx as nx

from epiforecast.observations import FullObservation
from epiforecast.data_assimilator import DataAssimilator

from epiforecast.populations import populate_ages, ClinicalStatistics, TransitionRates, king_county_transition_rates
from epiforecast.samplers import GammaSampler, AgeAwareBetaSampler

np.random.seed(10)
population = 100
n_status = 5
noise_var = 0.01 # independent variance on observations
n_samples = 2 # minimum number for an 'ensemble'

observations = FullObservation(population, noise_var, "Full state observation 1% noise")
assimilator = DataAssimilator(observations = observations, errors = [])

# Generate random current state
current_state = np.random.uniform(0.0, 1.0/6.0, (n_samples, population * n_status))

print(np.sum(current_state[:, 1::population], axis=0))

transmission_rate = np.random.uniform(0.03, 0.1, (n_samples, 1))
transition_rates = []

for i in range(n_samples):
    transition_rates.append(king_county_transition_rates(population))
        
# Some data (corresponding to the size of the current state)
synthetic_data = 1.0/6.0 * np.ones(population * n_status)

# currently no implemented Observation classes rely upon this.
contact_network = nx.watts_strogatz_graph(population, 12, 0.1, 1)

for i in np.arange(20):
    # Update states, the transition_rates object and the transmission rate array.
    new_state, new_transition_rates, new_transmission_rate = assimilator.update(current_state, synthetic_data,
                                                                                  full_ensemble_transition_rates = transition_rates,
                                                                                 full_ensemble_transmission_rate = transmission_rate,
                                                                                                 contact_network = contact_network
                                                                                )
    transmission_rate = new_transmission_rate
