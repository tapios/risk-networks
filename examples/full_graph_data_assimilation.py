import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

# Utilities for generating random populations
from epiforecast.populations import assign_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.measurements import Observation

# Both numpy.random and random are used by the KineticModel.
np.random.seed(123)
random.seed(123)

#
# Create an example network
#
# define the contact network [networkx.graph] - haven't done this yet
contact_network = nx.watts_strogatz_graph(100, 12, 0.1, 1)
population = len(contact_network)


#
# Clinical parameters of an age-distributed population
#

assign_ages(contact_network, distribution=[0.21, 0.4, 0.25, 0.08, 0.06])

#### Generate the synthetic data
transition_rates_truth = TransitionRates(contact_network,
                                        latent_periods = 3.7,
                                        community_infection_periods = 3.2,
                                        hospital_infection_periods = 5.0,
                                        hospitalization_fraction = AgeDependentBetaSampler(mean=[0.002,  0.01,   0.04, 0.075,  0.16], b=4),
                                        community_mortality_fraction = AgeDependentBetaSampler(mean=[ 1e-4,  1e-3,  0.003,  0.01,  0.02], b=4),
                                        hospital_mortality_fraction = AgeDependentBetaSampler(mean=[0.019, 0.075,  0.195, 0.328, 0.514], b=4)
)

#set transmission_rates
community_transmission_rate_truth = 12.0
hospital_transmission_reduction = 0.1

#### Generate synthetic data (with kinetic model)

synthetic_data = {i : 'I'  for i in range(population)}

#### Generate the ensemble parameters

ensemble_size = 10 # minimum number for an 'ensemble'



# We process the clinical data to determine transition rates between each epidemiological state,
transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(
        TransitionRates(contact_network,
                        latent_periods = np.random.normal(3.7,0.37),
                        community_infection_periods = np.random.normal(3.2,0.32),
                        hospital_infection_periods = np.random.normal(5.0,0.5),
                        hospitalization_fraction = AgeDependentBetaSampler(mean=[0.002,  0.01,   0.04, 0.075,  0.16], b=4),
                        community_mortality_fraction = AgeDependentBetaSampler(mean=[ 1e-4,  1e-3,  0.003,  0.01,  0.02], b=4),
                        hospital_mortality_fraction = AgeDependentBetaSampler(mean=[0.019, 0.075,  0.195, 0.328, 0.514], b=4)
                        )
                                    )

#set transmission_rates
community_transmission_rate_ensemble = np.random.normal(12.0,1.0, size=(ensemble_size,1))
hospital_transmission_reduction = 0.1



master_eqn_ensemble = MasterEquationModelEnsemble(contact_network,
                                                  transition_rates_ensemble,
                                                  community_transmission_rate_ensemble,
                                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                  ensemble_size = ensemble_size)

I_perc = 0.01
states_ensemble = np.zeros([ensemble_size, 5 * population])
for mm, member in enumerate(master_eqn_ensemble.ensemble):
    infected = np.random.choice(population, replace = False, size = int(population * I_perc))
    E, I, H, R, D = np.zeros([5, population])
    S = np.ones(population,)
    I[infected] = 1.
    S[infected] = 0.

    states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))
####





medical_infection_test = Observation(N = population,
                                     obs_frac = 1.0,
                                     obs_status = 'I',
                                     obs_name = "0.25 < Infected(100%) < 0.75")

# give the data assimilator the methods for how to choose observed states
observations=[medical_infection_test]
# give the data assimilator which transition rates and transmission rate to assimilate
transition_rates_to_update_str=['latent_periods', 'hospitalization_fraction']
transmission_rate_to_update_flag=True

# create the assimilator
assimilator = DataAssimilator(observations = observations,
                              errors = [],
                              transition_rates_to_update_str= transition_rates_to_update_str,
                              transmission_rate_to_update_flag = transmission_rate_to_update_flag)

assimilation_length=1
assimilation_interval=1
for i in range(assimilation_length):

    # health_service.discharge_and_admit_patients(..) #modifies the contact network
    # contact_simulator.mean_contact_rates(...)#generates new contact network weights
    # kinetic_model.set_contact_network(...)
    # master_eqn_model.set_contact_network(...)
    # synthetic_data[i] = kinetic model.simulate(..)

    res = master_eqn_ensemble.simulate(states_ensemble, assimilation_interval, n_steps = 10)
    states_ensemble = res["states"][:,:,-1]

    states_ensemble, transition_rates_ensemble, transmission_rate_ensemble = assimilator.update(states_ensemble,
                                                                                                synthetic_data,
                                                                                                full_ensemble_transition_rates = transition_rates_ensemble,
                                                                                                full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                                                                                user_network = contact_network)

    master_eqn_ensemble.update_transition_rates(transition_rates_ensemble)
    master_eqn_ensemble.update_transmission_rate(transmission_rate_ensemble)
