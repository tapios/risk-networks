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

from epiforecast.contact_simulator import DiurnalContactInceptionRate
from epiforecast.fast_contact_simulator import FastContactSimulator, DiurnalMeanContactRate
from epiforecast.kinetic_model_simulator import KineticModel, print_statuses
from epiforecast.scenarios import load_edges, random_epidemic

from epiforecast.node_identifier_helper import load_node_identifiers

from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService

from epiforecast.utilities import seed_numba_random_state

def simulation_average(model_data, sampling_time_step = 1):
    """
    Returns simulation data averages of simulation data and corresponding sampling times.
    """
    
    simulation_data_average = {}
    simulation_average = {}

    for key in model_data.statuses.keys():
        simulation_data_average[key] = []
        simulation_average[key] = []
    
    tav = 0

    sampling_times = []

    for i in range(len(model_data.times)):
        for key in model_data.statuses.keys():
            simulation_data_average[key].append(model_data.statuses[key][i])

        if model_data.times[i] >= tav:
            for key in model_data.statuses.keys():
                simulation_average[key].append(np.mean(simulation_data_average[key]))
                simulation_data_average[key] = []
            
            sampling_times.append(tav)
            tav += sampling_time_step

    return simulation_average, sampling_times
        
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

edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3_nobeds.txt')) 
node_identifiers = load_node_identifiers(os.path.join('..', 'data', 'networks', 'node_identifier_SBM_1e3_nobeds.txt'))

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)

#
# Build the contact simulator
#

contact_simulator = FastContactSimulator(

             n_contacts = nx.number_of_edges(contact_network),
    mean_event_duration = 0.6 / 60 / 24, # 1 minute in units of days
      mean_contact_rate = DiurnalMeanContactRate(minimum_i = 2, maximum_i = 22, minimum_j = 2, maximum_j = 22),
             start_time = -3 / 24, # negative start time allows short 'spinup' of contacts

)

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

static_contact_interval = 3 * hour

health_service = HealthService(patient_capacity = int(0.05 * len(contact_network)),
                               health_worker_population = len(node_identifiers['health_workers']),
                               static_population_network = contact_network)

epidemic_simulator = EpidemicSimulator(contact_network,            
                                                 mean_contact_lifetime = 0.5 * minute,
                                                contact_inception_rate = DiurnalContactInceptionRate(maximum=22, minimum=2),
                                                      transition_rates = transition_rates,
                                               static_contact_interval = static_contact_interval,
                                           community_transmission_rate = 12.0,
                                                        health_service = health_service,
                                       hospital_transmission_reduction = 0.1,
                                                       cycle_contacts = True)

statuses = random_epidemic(contact_network, fraction_infected=0.01)

epidemic_simulator.set_statuses(statuses)

synthetic_data = []
synthetic_data.append(statuses)

for i in range(int(1/static_contact_interval)):

    epidemic_simulator.run(stop_time = epidemic_simulator.time + static_contact_interval) # days

    statuses = epidemic_simulator.kinetic_model.current_statuses

    epidemic_simulator.set_statuses(statuses)

    synthetic_data.append(statuses)

print(synthetic_data)
#sampling_time_step = 1 # days

#Returns simulation data averages and corresponding sampling times

#simulation_data_average, sampling_times = simulation_average(kinetic_model, sampling_time_step=sampling_time_step)

#print(simulation_data_average, sampling_times)

#Produces synthetic data

# Create Data Assimilator,
# transition and transmission 'priors' for the ensemble
# create the master equations
# 

















#
# Plot the results and compare with NYC data.
#

np.savetxt("../data/simulation_data/simulation_data_NYC_DA_1e3.txt", np.c_[kinetic_model.times, kinetic_model.statuses['S'], kinetic_model.statuses['E'], kinetic_model.statuses['I'], kinetic_model.statuses['H'], kinetic_model.statuses['R'],kinetic_model.statuses['D']], header = 'S E I H R D seed: %d'%seed)

# # plot all model compartments
# fig, axs = plt.subplots(nrows=2, sharex=True)

# plt.sca(axs[0])
# plt.plot(kinetic_model.times, kinetic_model.statuses['S'])
# plt.ylabel("Total susceptible, $S$")

# plt.sca(axs[1])
# plt.plot(kinetic_model.times, kinetic_model.statuses['E'], label='Exposed')
# plt.plot(kinetic_model.times, kinetic_model.statuses['I'], label='Infected')
# plt.plot(kinetic_model.times, kinetic_model.statuses['H'], label='Hospitalized')
# plt.plot(kinetic_model.times, kinetic_model.statuses['R'], label='Resistant')
# plt.plot(kinetic_model.times, kinetic_model.statuses['D'], label='Deceased')

# plt.xlabel("Time (days)")
# plt.ylabel("Total $E, I, H, R, D$")
# plt.legend()

# image_path = ("../figs/simple_epidemic_with_slow_contact_simulator_" + 
#               "maxlambda_{:d}.png".format(contact_simulator.mean_contact_rate.maximum_i))

# print("Saving a visualization of results at", image_path)
# plt.savefig(image_path, dpi=480)
