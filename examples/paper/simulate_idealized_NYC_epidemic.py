import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import random 

from numba import set_num_threads

set_num_threads(1)

# Utilities for generating random populations
from epiforecast.populations import assign_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler, AgeDependentConstant

from epiforecast.kinetic_model_simulator import KineticModel, print_statuses
from epiforecast.scenarios import load_edges, random_epidemic

from epiforecast.node_identifier_helper import load_node_identifiers

from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService

from epiforecast.utilities import seed_numba_random_state, seed_three_random_states
        
#
# Set random seeds for reproducibility
#

seed = 942395

seed_three_random_states(seed)

#
# Load an example network
#

edges = load_edges(os.path.join('..', '..', 'data', 'networks', 'edge_list_SBM_1e5_nobeds.txt')) 
node_identifiers = load_node_identifiers(os.path.join('..', '..', 'data', 'networks', 'node_identifier_SBM_1e5_nobeds.txt'))

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)

#
# Clinical parameters of an age-distributed population
#

assign_ages(contact_network, distribution=[0.21, 0.4, 0.25, 0.08, 0.06])
                       
# We process the clinical data to determine transition rates between each epidemiological state,
transition_rates = TransitionRates(contact_network,

                  latent_periods = 3.7,
     community_infection_periods = 3.2,
      hospital_infection_periods = 5.0,
        hospitalization_fraction = AgeDependentConstant([0.002,  0.01,   0.04, 0.076,  0.16]),
    community_mortality_fraction = AgeDependentConstant([ 1e-4,  1e-3,  0.001,  0.07,  0.015]),
     hospital_mortality_fraction = AgeDependentConstant([0.019, 0.073,  0.193, 0.327, 0.512])

)

transmission_rate = 12.0
hospital_transmission_reduction = 0.1

# 
# Simulate the growth and equilibration of an epidemic
#

minute = 1 / 60 / 24
hour = 60 * minute

# Run the simulation

health_service = HealthService(health_workers = len(node_identifiers['health_workers']),
                               static_population_network = contact_network)

epidemic_simulator = EpidemicSimulator(contact_network,            
                                                 mean_contact_lifetime = 0.5 * minute,
                                                  day_inception_rate = 22,
                                                  night_inception_rate = 2,
                                                      transition_rates = transition_rates,
                                               static_contact_interval = 3 * hour,
                                           community_transmission_rate = 12.0,
                                                        health_service = health_service,
                                       hospital_transmission_reduction = 0.1)

statuses = random_epidemic(contact_network, fraction_infected=0.0025)

epidemic_simulator.set_statuses(statuses)

epidemic_simulator.run(stop_time = 18)

kinetic_model = epidemic_simulator.kinetic_model

# SD intervention

health_service = epidemic_simulator.health_service

statuses = epidemic_simulator.kinetic_model.current_statuses

epidemic_simulator = EpidemicSimulator(contact_network,            
                                                 mean_contact_lifetime = 0.5 * minute,
                                                  day_inception_rate = 8,
                                                  night_inception_rate = 2,
                                                      transition_rates = transition_rates,
                                               static_contact_interval = 3 * hour,
                                           community_transmission_rate = 12.0,
                                                        health_service = health_service,
                                       hospital_transmission_reduction = 0.1)

epidemic_simulator.set_statuses(statuses)

epidemic_simulator.run(stop_time = 83) # days

for key in kinetic_model.statuses.keys():
   kinetic_model.statuses[key].extend(epidemic_simulator.kinetic_model.statuses[key])


kinetic_model.times.extend(kinetic_model.times[-1]+epidemic_simulator.kinetic_model.times)

# loosening SD intervention

health_service = epidemic_simulator.health_service

statuses = epidemic_simulator.kinetic_model.current_statuses

#inf_cnt = 50
#cnt = 0

#for i in range(len(statuses)):
#    if statuses[i] == 'S' and cnt <= inf_cnt:
#        statuses[i] = 'I'
#        cnt += 1

epidemic_simulator = EpidemicSimulator(contact_network,            
                                                 mean_contact_lifetime = 0.5 * minute,
                                                  day_inception_rate = 22,
                                                  night_inception_rate = 2,
                                                      transition_rates = transition_rates,
                                               static_contact_interval = 3 * hour,
                                           community_transmission_rate = 12.0,
                                                        health_service = health_service,
                                       hospital_transmission_reduction = 0.1)

epidemic_simulator.set_statuses(statuses)

epidemic_simulator.run(stop_time = 90) # days

for key in kinetic_model.statuses.keys():
   kinetic_model.statuses[key].extend(epidemic_simulator.kinetic_model.statuses[key])


kinetic_model.times.extend(kinetic_model.times[-1]+epidemic_simulator.kinetic_model.times)

#
# Plot the results and compare with NYC data.
#

np.savetxt("../../data/simulation_data/simulation_data_NYC_1e5.txt", np.c_[kinetic_model.times, kinetic_model.statuses['S'], kinetic_model.statuses['E'], kinetic_model.statuses['I'], kinetic_model.statuses['H'], kinetic_model.statuses['R'],kinetic_model.statuses['D']], header = 'S E I H R D seed: %d'%seed)

