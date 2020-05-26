import os, sys; sys.path.append(os.path.join(".."))

import numpy as np
import networkx as nx

from epiforecast.observations import FullObservation
from epiforecast.data_assimilator import DataAssimilator

from epiforecast.populations import populate_ages, ClinicalStatistics, TransitionRates 
#from epiforecast.populations import king_county_transition_rates
from epiforecast.samplers import GammaSampler, BetaSampler

np.random.seed(10)

population = 100
n_status = 5
noise_var = 0.01 # independent variance on observations
n_samples = 2 # minimum number for an 'ensemble'


transition_rates_to_update_str=['hospitalization_fraction']
observations = FullObservation(population, noise_var, "Full state observation 1% noise")
assimilator = DataAssimilator(observations = observations, errors = [], transition_rates_to_update_str= transition_rates_to_update_str)

# Generate random current state
current_state = np.random.uniform(0.0, 1.0/6.0, (n_samples, population * n_status))

print(np.sum(current_state[:, 1::population], axis=0))

transmission_rate = np.random.uniform(0.03, 0.1, (n_samples, 1))
transition_rates = []

age_distribution = [ 0.23,  # 0-19 years
                     0.39,  # 20-44 years
                     0.25,  # 45-64 years
                     0.079  # 65-75 years
                    ]

# 75 onwards
age_distribution.append(1 - sum(age_distribution))

ages = populate_ages(population, distribution=age_distribution)


for i in range(n_samples):
    
    # Next, we randomly generate clinical properties for our example population.
    # Note that the units of 'periods' are days, and the units of 'rates' are 1/day.
    latent_periods = ClinicalStatistics(ages = ages, minimum = 2,
                                                 sampler = GammaSampler(k=1.7, theta=2.0))

    community_infection_periods = ClinicalStatistics(ages = ages, minimum = 1,
                                                 sampler = GammaSampler(k=1.5, theta=2.0))

    hospital_infection_periods = ClinicalStatistics(ages = ages, minimum = 1,
                                                sampler = GammaSampler(k=1.5, theta=3.0))

    hospitalization_fraction = ClinicalStatistics(ages = [0.5],
    sampler = BetaSampler(mean=0.2, b=4)) #single value broadcast to all nodes
 
    community_mortality_fraction = ClinicalStatistics(ages = [0.5],
    sampler = BetaSampler(mean=0.01, b=4))

    hospital_mortality_fraction  = ClinicalStatistics(ages = [0.5],
    sampler = BetaSampler(mean=0.01, b=4))

    #test only 1 latent period
    t_rates = TransitionRates(population,
                              latent_periods,
                              community_infection_periods,
                              hospital_infection_periods,
                              hospitalization_fraction,
                              community_mortality_fraction,
                              hospital_mortality_fraction)

    transition_rates.append(t_rates)
    
    #    transition_rates.append(king_county_transition_rates(population))
        
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
