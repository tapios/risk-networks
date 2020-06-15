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

from epiforecast.populations import assign_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler, AgeDependentConstant

from epiforecast.scenarios import load_edges, random_epidemic

from epiforecast.epiplots import plot_ensemble_states, plot_kinetic_model_data, plot_scalar_parameters

from epiforecast.node_identifier_helper import load_node_identifiers
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.measurements import Observation, DataObservation, DataNodeObservation
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.utilities import seed_numba_random_state
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries

def deterministic_risk(contact_network, initial_states, ensemble_size=1):

    population = len(contact_network)
    states_ensemble = np.zeros([ensemble_size, 5 * population])

    init_catalog = {'S': False, 'I': True}
    infected = np.array([init_catalog[status] for status in list(initial_states.values())])

    for mm in range(ensemble_size):
        E, I, H, R, D = np.zeros([5, population])
        S = np.ones(population,)
        I[infected] = 1.
        S[infected] = 0.

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

    return states_ensemble

#
# Set random seeds for reproducibility
#

# Both numpy.random and random are used by the KineticModel.
seed = 212212

np.random.seed(seed)
random.seed(seed)

# set numba seed

seed_numba_random_state(seed)

#
# Load an example network
#




edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3_nobeds.txt'))
node_identifiers = load_node_identifiers(os.path.join('..', 'data', 'networks', 'node_identifier_SBM_1e3_nobeds.txt'))

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)

#
# Build the contact simulator
#
start_time =0.0

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

#
# Simulate the growth and equilibration of an epidemic
#
static_contact_interval = 3 * hour
simulation_length = 30

print("We first create an epidemic for",simulation_length,"days, then we solve the master equations forward and backward")
health_service = HealthService(static_population_network = contact_network,
                               health_workers = node_identifiers['health_workers'],
                               health_workers_per_patient=5)

mean_contact_lifetime=0.5*minute
hospital_transmission_reduction = 0.1

epidemic_simulator = EpidemicSimulator(
                 contact_network = contact_network,
                transition_rates = transition_rates,
     community_transmission_rate = community_transmission_rate,
 hospital_transmission_reduction = hospital_transmission_reduction,
         static_contact_interval = static_contact_interval,
           mean_contact_lifetime = mean_contact_lifetime,
              day_inception_rate = 22,
            night_inception_rate = 2,
                  health_service = health_service,
                      start_time = start_time
                                      )
ensemble_size = 2 # minimum number for an 'ensemble'

transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(transition_rates)

#set transmission_rates
community_transmission_rate_ensemble = community_transmission_rate*np.ones([ensemble_size,1]) 


master_eqn_ensemble = MasterEquationModelEnsemble(contact_network = contact_network,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = community_transmission_rate_ensemble,
                                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                  ensemble_size = ensemble_size,
                                                  start_time = start_time)

# Create storage for networks and data
epidemic_data_storage = StaticIntervalDataSeries(static_contact_interval)

time = start_time

statuses = random_epidemic(contact_network,
                           fraction_infected=0.01)

states_ensemble = deterministic_risk(contact_network,
                                     statuses,
                                     ensemble_size = ensemble_size)


epidemic_simulator.set_statuses(statuses)
master_eqn_ensemble.set_states_ensemble(states_ensemble)

fig, axes = plt.subplots(1, 3, figsize = (16, 4))
# First we run and save the epidemic
for i in range(int(simulation_length/static_contact_interval)):
    
    epidemic_simulator.run(stop_time = epidemic_simulator.time + static_contact_interval)
    #save the start time network and statuses
    epidemic_data_storage.save_network_by_start_time(contact_network=contact_network, start_time=time)
    epidemic_data_storage.save_start_statuses_to_network(start_time=time, start_statuses=statuses)

    #update the statuses and time
    statuses = epidemic_simulator.kinetic_model.current_statuses
    time=epidemic_simulator.time

    #save the statuses at the new time
    epidemic_data_storage.save_end_statuses_to_network(end_time=time, end_statuses=statuses)
    
    axes = plot_kinetic_model_data(epidemic_simulator.kinetic_model,
                                   axes = axes)

    plt.savefig('kinetic_and_master.png', rasterized=True, dpi=150)



time = start_time

# Then we run the master equations forward on the loaded networks
for i in range(int(simulation_length/static_contact_interval)):

    loaded_static_network=epidemic_data_storage.get_network_from_start_time(start_time=time)
    loaded_contact_network= loaded_static_network.contact_network
    master_eqn_ensemble.set_contact_network_and_contact_duration(loaded_contact_network) #do not need to reset weights as already set in kinetic model
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)
    
    #at the update the time
    time = time + static_contact_interval
    master_eqn_ensemble.set_states_ensemble(states_ensemble)

    axes = plot_ensemble_states(master_eqn_ensemble.states_trace,
                                master_eqn_ensemble.simulation_time,
                                axes = axes,
                                xlims = (-0.1, simulation_length),
                                a_min = 0.0)
    
    plt.savefig('kinetic_and_master.png', rasterized=True, dpi=150)

master_eqn_ensemble = MasterEquationModelEnsemble(contact_network = contact_network,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = community_transmission_rate_ensemble,
                                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                  ensemble_size = ensemble_size,
                                                  start_time = time)

master_eqn_ensemble.set_states_ensemble(states_ensemble)

#then we run the master equations backwards on the loaded networks
for i in range(int(simulation_length/static_contact_interval)):

        loaded_static_network=epidemic_data_storage.get_network_from_end_time(end_time=time)
        loaded_contact_network= loaded_static_network.contact_network
        master_eqn_ensemble.set_contact_network(loaded_contact_network) #do not need to reset weights as already set in kinetic model
        states_ensemble = master_eqn_ensemble.simulate_backwards(static_contact_interval, n_steps = 100)

        #at the update the time
        time = time - static_contact_interval
        master_eqn_ensemble.set_states_ensemble(states_ensemble)

        axes = plot_ensemble_states(master_eqn_ensemble.states_trace,
                                    master_eqn_ensemble.simulation_time,
                                    axes = axes,
                                    xlims = (-0.1, simulation_length),
                                    a_min = 0.0)
        
        plt.savefig('kinetic_and_master.png', rasterized=True, dpi=150)
