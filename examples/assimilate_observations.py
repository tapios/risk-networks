import os, sys; sys.path.append(os.path.join(".."))

import numpy as np
import networkx as nx

from epiforecast.observations import FullObservation
from epiforecast.data_assimilator import DataAssimilator

from epiforecast.populations import populate_ages, ClinicalStatistics, TransitionRates
from epiforecast.samplers import GammaSampler, AgeAwareBetaSampler

np.random.seed(10)
population = 100
n_status = 5
noise_var = 0.01 # independent variance on observations
n_samples = 2 # minimum number for an 'ensemble'

#Determine which rates to assimilate




#transition_rates_to_update_str = ['latent_periods',
#                                  'community_infection_periods',
#                                  'hospital_infection_periods',
#                                  'hospitalization_fraction', 
#                                  'community_mortality_fraction',
#                                  'hospital_mortality_fraction']            
                                  
transition_rates_to_update_str=['community_infection_periods','community_mortality_fraction']                                  
transmission_rate_to_update_flag = True

assimilator = DataAssimilator(observations = FullObservation(population,noise_var,"Full state observation 1% noise"), 
                              errors = [],
                              transition_rates_to_update_str=transition_rates_to_update_str,
                              transmission_rate_to_update_flag=transmission_rate_to_update_flag)

# current state, transition_rates (20 of them), transmission rate (1)
current_state = np.random.uniform(0.0, 1.0/6.0, (n_samples, population * n_status))
print(np.sum(current_state[:,1::population],axis=0))

# For transition_rates see generate_clinical_statistics.py

age_distribution = [ 1. ]
ages = populate_ages(population, distribution=age_distribution)

transition_rates=[]
for i in range(n_samples):

    latent_periods = ClinicalStatistics(ages = ages, minimum = 2,
                                        sampler = GammaSampler(k=1.7, theta=2.0))

    community_infection_periods = ClinicalStatistics(ages = ages, minimum = 1,
                                                     sampler = GammaSampler(k=1.5, theta=2.0))

    hospital_infection_periods = ClinicalStatistics(ages = ages, minimum = 1,
                                                    sampler = GammaSampler(k=1.5, theta=3.0))

    hospitalization_fraction = ClinicalStatistics(ages = ages,
                                                  sampler = AgeAwareBetaSampler(mean=[ 0.25 ], b=4))

    community_mortality_fraction = ClinicalStatistics(ages = ages,
                                                      sampler = AgeAwareBetaSampler(mean=[0.02], b=4))

    hospital_mortality_fraction  = ClinicalStatistics(ages = ages,
                                                      sampler = AgeAwareBetaSampler(mean=[0.04], b=4))

    t_rates = TransitionRates(latent_periods,
                              community_infection_periods,
                              hospital_infection_periods,
                              hospitalization_fraction,
                              community_mortality_fraction,
                              hospital_mortality_fraction)

    transition_rates.append(t_rates) 
    

transmission_rate = np.random.uniform(0.03, 0.1, (n_samples,1))
 
# some data (corresponding to the size of the current state)
synthetic_data = 1.0/6.0 * np.ones(population * n_status)

# currently no implemented Observation classes rely upon this.
contact_network=nx.watts_strogatz_graph(population,12,0.1,1)


for i in np.arange(20):
    #update states, the transition_rates object and the transmission rate array.
    new_state, new_transition_rates, new_transmission_rate = assimilator.update(current_state,
                                                                                 synthetic_data,
                                                                                 full_ensemble_transition_rates=transition_rates,
                                                                                 full_ensemble_transmission_rate=transmission_rate,
                                                                                 contact_network=contact_network)
    transmission_rate=new_transmission_rate
    
