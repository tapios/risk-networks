import os, sys; sys.path.append(os.path.join(".."))

import numpy as np

from epiforecast.observations import FullObservation
from epiforecast.data_assimilator import DataAssimilator

population = 100
n_status = 5
noise_var = 0.01 # independent variance on observations
n_samples = 2 # minimum number for an 'ensemble'

#Determine which rates to assimilate




transition_rates_to_update_str = ['latent_period',
                                  'community_infection_periods',
                                  'hospital_infection_periods',
                                  'hospitalization_fraction',
                                  'community_mortality_fraction',
                                  'hospital_mortality_fraction']                              
transmission_rate_to_update_flag = True

assimilator = DataAssimilator(observations = FullObservation(population,noise_var,"Full state observation 1% noise"), 
                              errors = [],
                              transition_rates_to_update_str=transition_rates_to_update_str,
                              transmission_rate_to_update_flag=transmission_rate_to_update_flag)

# current state, transition_rates (20 of them), transmission rate (1)
current_state = np.random.uniform(0.0, 1.0/6.0, (n_samples, population * n_status))
print(np.sum(current_state[:,1::population],axis=0))

# For transition_rates see generate_clinical_statistics.py
transition_rates = transition_rates(...)
transmission_rates = np.random.uniform(0.01, 0.5, (n_samples,1))

# some data (corresponding to the size of the current state)
synthetic_data = 1.0/6.0 * np.ones(population * n_status)

# currently no implemented Observation classes rely upon this.
contact_network=[]

#update states, the transition_rates object and the transmission rate array.
new_state, new_transition_rates, new_transmission_rates = assimilator.update(current_state,
                                                                             synthetic_data,
                                                                             full_ensemble_transition_rates=transition_rates,
                                                                             full_ensemble_transmission_rates=transmission_rates
                                                                             contact_network=contact_network)

