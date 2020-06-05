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

from epiforecast.fast_contact_simulator import FastContactSimulator, DiurnalMeanContactRate
from epiforecast.kinetic_model_simulator import KineticModel, print_statuses
from epiforecast.scenarios import load_edges, random_epidemic

from epiforecast.epiplots import plot_master_eqns

from epiforecast.node_identifier_helper import load_node_identifiers
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.measurements import Observation
from epiforecast.data_assimilator import DataAssimilator

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
start_time = -3 / 24

minute = 1 / 60 / 24
hour = 60 * minute


contact_simulator = FastContactSimulator(

             n_contacts = nx.number_of_edges(contact_network),
    mean_event_duration = 0.5*minute, # 1 minute in units of days
      mean_contact_rate = DiurnalMeanContactRate(minimum_i = 2, maximum_i = 22, minimum_j = 2, maximum_j = 22),
             start_time = start_time # negative start time allows short 'spinup' of contacts
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

community_transmission_rate = 12.0
hospital_transmission_reduction = 0.1

#
# Simulate the growth and equilibration of an epidemic
#
static_contact_interval = 12 * hour

health_service = HealthService(patient_capacity = int(0.05 * len(contact_network)),
                               health_worker_population = len(node_identifiers['health_workers']),
                               static_population_network = contact_network)



max_edges = nx.number_of_edges(contact_network) + 5 * health_service.patient_capacity

kinetic_model = KineticModel(contact_network = contact_network,
                             transition_rates = transition_rates,
                             community_transmission_rate = community_transmission_rate,
                             hospital_transmission_reduction = hospital_transmission_reduction)

## construct master equations model

ensemble_size = 100 # minimum number for an 'ensemble'

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

master_eqn_ensemble = MasterEquationModelEnsemble(contact_network = contact_network,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = community_transmission_rate_ensemble,
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

time = start_time

statuses = random_epidemic(contact_network, fraction_infected=0.01)

kinetic_model.set_statuses(statuses)
master_eqn_ensemble.set_states_ensemble(states_ensemble)

# print(static_contact_interval)
# print(int(simulation_length/static_contact_interval))

fig, axes = plt.subplots(1, 2, figsize = (15, 5))

for i in range(int(simulation_length/static_contact_interval)):

    # health_service discharge and admit patients [changes the contact network]
    health_service.discharge_and_admit_patients(kinetic_model.current_statuses,
                                                contact_network)
    # contact_simulator run [changes the mean contact duration on the given network]
    contact_duration = contact_simulator.mean_contact_duration(stop_time=time+static_contact_interval)

    # pass the averaged contact durations to the models
    kinetic_model.set_mean_contact_duration(contact_duration)
    master_eqn_ensemble.set_mean_contact_duration(contact_duration)
    # master_eqn_ensemble.set_mean_contact_duration(None)

    # run the models [kinetic produces the current statuses used as data]
    #                [master eqn produces the current states of the risk model]
    statuses = kinetic_model.simulate(static_contact_interval)
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval)

    # perform data assimlation [update the master eqn states, the transition rates, and the transmission rate (if supplied)]
    (states_ensemble,
     transition_rates_ensemble,
     community_transmission_rate_ensemble
     ) = assimilator.update(states_ensemble,
                            statuses,
                            full_ensemble_transition_rates = transition_rates_ensemble,
                            full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                            user_network = contact_network)

    #update model parameters (transition and transmission rates) of the master eqn model
    master_eqn_ensemble.update_transition_rates(transition_rates_ensemble)
    master_eqn_ensemble.update_transmission_rate(community_transmission_rate_ensemble)

    #update states/statuses/times for next iteration
    kinetic_model.set_statuses(statuses)
    master_eqn_ensemble.set_states_ensemble(states_ensemble)
    time += static_contact_interval
    # print("Current time",time)
    print("Current time",time)
    print(states_ensemble[0,population:2*population])

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
