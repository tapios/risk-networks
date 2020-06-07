import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import pandas as pd
import random
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from numba import set_num_threads

set_num_threads(1)

# Utilities for generating random populations
from epiforecast.populations import assign_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler, AgeDependentConstant

from epiforecast.scenarios import load_edges, random_epidemic

from epiforecast.epiplots import plot_master_eqns

from epiforecast.node_identifier_helper import load_node_identifiers
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.measurements import Observation
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.user_base import ContiguousUserBase, contiguous_indicators
from epiforecast.utilities import seed_numba_random_state





def random_risk(contact_network, fraction_infected = 0.01, ensemble_size=1):
    
    user_population = len(contact_network)
    states_ensemble = np.zeros([ensemble_size, 5 * user_population])
    for mm in range(ensemble_size):
        infected = np.random.choice(user_population, replace = False, size = int(user_population * fraction_infected))
        E, I, H, R, D = np.zeros([5, user_population])
        S = np.ones(user_population,)
        I[infected] = 1.
        S[infected] = 0.

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

    return states_ensemble


#
# Set random seeds for reproducibility
#

# Both numpy.random and random are used by the KineticModel.
seed = 2132

np.random.seed(seed)
random.seed(seed)

# set numba seed

seed_numba_random_state(seed)

#
# Load an example network
#

edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e4_nobeds.txt'))
node_identifiers = load_node_identifiers(os.path.join('..', 'data', 'networks', 'node_identifier_SBM_1e4_nobeds.txt'))

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)

#
# Build the contact simulator
#
start_time = -3 / 24

minute = 1 / 60 / 24
hour = 60 * minute

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

community_transmission_rate = 12.0
hospital_transmission_reduction = 0.1

#
# Simulate the growth and equilibration of an epidemic
#
static_contact_interval = 3 * hour

health_service = HealthService(patient_capacity = int(0.05 * len(contact_network)),
                               health_worker_population = len(node_identifiers['health_workers']),
                               static_population_network = contact_network)


mean_contact_lifetime=0.5*minute

epidemic_simulator = EpidemicSimulator( 
                 contact_network = contact_network,
                transition_rates = transition_rates,
         static_contact_interval = static_contact_interval,
     community_transmission_rate = community_transmission_rate,
 hospital_transmission_reduction = hospital_transmission_reduction,
           mean_contact_lifetime = mean_contact_lifetime,
              day_inception_rate = 22,
            night_inception_rate = 2,
                  health_service = health_service,
                      start_time = start_time
                                      )


### Create the user base.
user_fraction=0.05
user_base = ContiguousUserBase(contact_network,
                               user_fraction,
                               method="neighbor",
                               seed_user=None)

user_population=len(user_base.contact_network)
print("size of network", population,
      " and size of user base", user_population)

#some info about the user base:
interior,boundary,mean_exterior_neighbors = contiguous_indicators(contact_network,user_base.contact_network)
print("user base interior nodes:", interior)
print("user base boundary nodes:", boundary)
print("average exterior neighbors of boundary node:", mean_exterior_neighbors)


## construct master equations model

ensemble_size = 100 # minimum number for an 'ensemble'

# We process the clinical data to determine transition rates between each epidemiological state,
transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(
        TransitionRates(user_base.contact_network,
                        latent_periods = np.random.normal(3.7,0.37),
                        community_infection_periods = np.random.normal(3.2,0.32),
                        hospital_infection_periods = np.random.normal(5.0,0.5),
                        hospitalization_fraction = AgeDependentBetaSampler(mean=[0.002,  0.01,   0.04, 0.075,  0.16], b=4),
                        community_mortality_fraction = AgeDependentBetaSampler(mean=[ 1e-4,  1e-3,  0.003,  0.01,  0.02], b=4),
                        hospital_mortality_fraction = AgeDependentBetaSampler(mean=[0.019, 0.075,  0.195, 0.328, 0.514], b=4)
                        )
        )
#set transmission_rates
community_transmission_rate_ensemble = np.random.normal(12.0, 1.0, size=(ensemble_size,1))
exogenous_transmission_rate_ensemble = np.random.normal(2.0,  1.0, size=(ensemble_size,1))

master_eqn_ensemble = MasterEquationModelEnsemble(contact_network = user_base.contact_network,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = community_transmission_rate_ensemble,
                                        exogenous_transmission_rate = exogenous_transmission_rate_ensemble,
                                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                  ensemble_size = ensemble_size)
####

medical_infection_test = Observation(N = user_population,
                                     obs_frac = 1.0,
                                     obs_status = 'I',
                                     obs_name = "0.25 < Infected(100%) < 0.5",
                                     min_threshold=0.1,
                                     max_threshold=0.7)

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

simulation_length = 30 #Number of days

#initial conditions
statuses = random_epidemic(contact_network,
                           fraction_infected = 0.01)
states_ensemble = random_risk(user_base.contact_network,
                              fraction_infected = 0.01,
                              ensemble_size = ensemble_size)
epidemic_simulator.set_statuses(statuses)
master_eqn_ensemble.set_states_ensemble(states_ensemble)

fig, axes = plt.subplots(1, 2, figsize = (15, 5))

for i in range(int(simulation_length/static_contact_interval)):

    epidemic_simulator.run(stop_time = epidemic_simulator.time + static_contact_interval)
    #Within the epidemic_simulator:
    # health_service discharge and admit patients [changes the contact network]
    # contact_simulator run [changes the mean contact duration on the given network]
    # set the new contact rates on the network! kinetic_model.set_mean_contact_duration(contact_duration)
    # run the kinetic model [kinetic produces the current statuses used as data]

    # as kinetic sets the weights, we do not need to update the contact network.
    # run the master equation model [master eqn produces the current states of the risk model]
    master_eqn_ensemble.set_mean_contact_duration() #do not need to reset weights as already set in kinetic model
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval)

    # perform data assimlation [update the master eqn states, the transition rates, and the transmission rate (if supplied)]
    (states_ensemble,
     transition_rates_ensemble,
     community_transmission_rate_ensemble
     ) = assimilator.update(states_ensemble,
                            epidemic_simulator.kinetic_model.current_statuses,
                            full_ensemble_transition_rates = transition_rates_ensemble,
                            full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                            user_network = user_base.contact_network)

    #update model parameters (transition and transmission rates) of the master eqn model
    master_eqn_ensemble.update_transition_rates(transition_rates_ensemble)
    master_eqn_ensemble.update_transmission_rate(community_transmission_rate_ensemble)

    #update states/statuses/times for next iteration
    master_eqn_ensemble.set_states_ensemble(states_ensemble)
        
    print("Current time",epidemic_simulator.time)
    #print(states_ensemble[0,population:2*population])

#
# Plot the results and compare with NYC data.
#

np.savetxt("../data/simulation_data/simulation_data_NYC_DA_1e3.txt", np.c_[epidemic_simuator.kinetic_model.times, epidemic_simuator.kinetic_model.statuses['S'], epidemic_simuator.kinetic_model.statuses['E'], epidemic_simuator.kinetic_model.statuses['I'], epidemic_simuator.kinetic_model.statuses['H'], epidemic_simuator.kinetic_model.statuses['R'],epidemic_simuator.kinetic_model.statuses['D']], header = 'S E I H R D seed: %d'%seed)
