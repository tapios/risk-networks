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
# Utilities for generating random populations
from epiforecast.populations import assign_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler, AgeDependentConstant

from epiforecast.contact_simulator import DiurnalContactInceptionRate

from epiforecast.scenarios import load_edges, random_epidemic

from epiforecast.epiplots import plot_ensemble_states, plot_kinetic_model_data, plot_ensemble_transmission_latent_fraction


from epiforecast.node_identifier_helper import load_node_identifiers
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.measurements import Observation, DataObservation, DataNodeObservation
from epiforecast.data_assimilator import DataAssimilator

from epiforecast.utilities import seed_numba_random_state

def random_risk(contact_network, fraction_infected = 0.01, ensemble_size=1):

    population = len(contact_network)
    states_ensemble = np.zeros([ensemble_size, 5 * population])
    for mm in range(ensemble_size):
        infected = np.random.choice(population, replace = False, size = int(population * fraction_infected))
        E, I, H, R, D = np.zeros([5, population])
        S = np.ones(population,)
        I[infected] = 1.
        S[infected] = 0.

        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

    return states_ensemble

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
seed = 213213

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
static_contact_interval = 6 * hour
simulation_length = 30

health_service = HealthService(patient_capacity = int(0.05 * len(contact_network)),
                               health_worker_population = len(node_identifiers['health_workers']),
                               static_population_network = contact_network)



mean_contact_lifetime=0.5*minute

epidemic_simulator = EpidemicSimulator(
                 contact_network = contact_network,
           mean_contact_lifetime = mean_contact_lifetime,
          contact_inception_rate = DiurnalContactInceptionRate(minimum = 2, maximum = 22),
                transition_rates = transition_rates,
     community_transmission_rate = community_transmission_rate,
 hospital_transmission_reduction = hospital_transmission_reduction,
         static_contact_interval = static_contact_interval,
                  health_service = health_service,
                      start_time = start_time
                                      )
ensemble_size = 20 #100 # minimum number for an 'ensemble'

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

####

medical_infection_test = Observation(N = population,
                                     obs_frac = 1.00,
                                     obs_status = 'I',
                                     obs_name = "0.25 < Infected(100%) < 0.75",
                                     min_threshold=0.00,
                                     max_threshold=1.00)

random_infection_test = Observation(N = population,
                                     obs_frac = 0.01,
                                     obs_status = 'I',
                                     obs_name = "Random Infection Test")

hospital_records = DataObservation(N = population,
                                   bool_type=True,
                                   obs_status = 'H',
                                   obs_name = "Hospitalized from Data",
                                   specificity  = 0.999,
                                   sensitivity  = 0.999)

death_records = DataObservation(N = population,
                                bool_type=True,
                                obs_status = 'D',
                                obs_name = "Deceased from Data",
                                specificity  = 0.999,
                                sensitivity  = 0.999)


# give the data assimilator the methods for how to choose observed states
# observations=[medical_infection_test, random_infection_test, hospital_records, death_records]
# observations=[medical_infection_test]
observations=[hospital_records, death_records]

# give the data assimilator which transition rates and transmission rate to assimilate
transition_rates_to_update_str=['latent_periods', 'hospitalization_fraction']
transmission_rate_to_update_flag=True

# create the assimilator
assimilator = DataAssimilator(observations = observations,
                              errors = [],
                              transition_rates_to_update_str= transition_rates_to_update_str,
                              transmission_rate_to_update_flag = transmission_rate_to_update_flag)



time = start_time

statuses = random_epidemic(contact_network,
                           fraction_infected=0.01)

# states_ensemble = random_risk(contact_network,
#                               fraction_infected = 0.01,
#                               ensemble_size = ensemble_size)

states_ensemble = deterministic_risk(contact_network,
                              statuses,
                              ensemble_size = ensemble_size)

epidemic_simulator.set_statuses(statuses)
master_eqn_ensemble.set_states_ensemble(states_ensemble)

# print(static_contact_interval)
# print(int(simulation_length/static_contact_interval))

# fig, axes = plt.subplots(1, 2, figsize = (15, 5))
fig, axes = plt.subplots(1, 3, figsize = (16, 4))

community_transmission_rate_trace = np.copy(community_transmission_rate_ensemble)
latent_periods_trace              = np.copy(np.array([member.latent_periods for member in transition_rates_ensemble]).reshape(-1,1))

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
    # would love to double check this! ^
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)

    if i % 1 == 0:
    # perform data assimlation [update the master eqn states, the transition rates, and the transmission rate (if supplied)]
        (states_ensemble,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
         ) = assimilator.update(ensemble_state = states_ensemble,
                                data = epidemic_simulator.kinetic_model.current_statuses,
                                full_ensemble_transition_rates = transition_rates_ensemble,
                                full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                user_network = contact_network)

        #update model parameters (transition and transmission rates) of the master eqn model
        master_eqn_ensemble.update_transition_rates(transition_rates_ensemble)
        master_eqn_ensemble.update_transmission_rate(community_transmission_rate_ensemble)


        # for tracking purposes
        community_transmission_rate_trace = np.hstack([community_transmission_rate_trace, community_transmission_rate_ensemble])
        latent_periods_trace              = np.hstack([latent_periods_trace, np.array([member.latent_periods for member in transition_rates_ensemble]).reshape(-1,1)])

    #update states/statuses/times for next iteration
    master_eqn_ensemble.set_states_ensemble(states_ensemble)

    axes = plot_ensemble_states(master_eqn_ensemble.states_trace,
                                master_eqn_ensemble.simulation_time,
                                axes = axes,
                                xlims = (-0.1, simulation_length),
                                a_min = 0.0)

    axes = plot_kinetic_model_data(epidemic_simulator.kinetic_model,
                                   axes = axes)

    plt.savefig('da_dic_tprobs_ninfectest_wnodehospital_wnodedeath_nrandtest.png', rasterized=True, dpi=150)


# time_horizon      = np.linspace(0.0, simulation_length, int(simulation_length/static_contact_interval) + 1)
# axes = plot_ensemble_transmission_latent_fraction(community_transmission_rate_trace, latent_periods_trace, time_horizon)
# plt.savefig('da_parameters_deterministic_ic.png', rasterized=True, dpi=150)


# np.savetxt("../data/simulation_data/simulation_data_NYC_DA_1e3.txt", np.c_[kinetic_model.times, kinetic_model.statuses['S'], kinetic_model.statuses['E'], kinetic_model.statuses['I'], kinetic_model.statuses['H'], kinetic_model.statuses['R'],kinetic_model.statuses['D']], header = 'S E I H R D seed: %d'%seed)

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
