import os, sys; sys.path.append(os.path.join(".."))
import pdb
import copy

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

    init_catalog = {'S': False, 'E': False, 'I': True, 'R': False, 'H': False, 'D': False}
    infected = np.array([init_catalog[status] for status in list(initial_states.values())])

    init_catalog = {'S': False, 'E': True, 'I': False, 'R': False, 'H': False, 'D': False}
    exposed = np.array([init_catalog[status] for status in list(initial_states.values())])

    init_catalog = {'S': False, 'E': False, 'I': False, 'R': True, 'H': False, 'D': False}
    resistant = np.array([init_catalog[status] for status in list(initial_states.values())])

    init_catalog = {'S': False, 'E': False, 'I': False, 'R': False, 'H': True, 'D': False}
    hospitalized = np.array([init_catalog[status] for status in list(initial_states.values())])

    init_catalog = {'S': False, 'E': False, 'I': False, 'R': False, 'H': False, 'D': True}
    dead = np.array([init_catalog[status] for status in list(initial_states.values())])

    for mm in range(ensemble_size):
        E, I, H, R, D = np.zeros([5, population])
        S = np.ones(population,)
        I[infected] = 1.
        S[infected] = 0.
        E[exposed] = 1.
        S[exposed] = 0.
        R[resistant] = 1.
        S[resistant] = 0.
        H[hospitalized] = 1.
        S[hospitalized] = 0.
        D[dead] = 1.
        S[dead] = 0.

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

#
# Simulate the growth and equilibration of an epidemic
#
static_contact_interval = 3 * hour
simulation_length = 30 

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
ensemble_size = 100 # minimum number for an 'ensemble'

transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(
 
        TransitionRates(contact_network,
                        latent_periods = np.random.normal(3.7,0.37),
                        community_infection_periods = np.random.normal(3.2, 0.32),
                        hospital_infection_periods  = np.random.normal(5.0, 0.50),
                        hospitalization_fraction    = transition_rates.hospitalization_fraction,
                        community_mortality_fraction = transition_rates.community_mortality_fraction,
                        hospital_mortality_fraction  = transition_rates.hospital_mortality_fraction
                       )
        )


#set transmission_rates
community_transmission_rate_ensemble = community_transmission_rate*np.ones([ensemble_size,1]) #np.random.normal(12.0,1.0, size=(ensemble_size,1))

master_eqn_ensemble = MasterEquationModelEnsemble(contact_network = contact_network,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = community_transmission_rate_ensemble,
                                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                  ensemble_size = ensemble_size,
                                                  start_time=start_time)

####
#possible observations:
#medical_infection_test = Observation(N = population,
#                                     obs_frac = 0.20,
#                                     obs_status = 'I',
#                                     obs_name = "0.01 < Infected(100%) < 0.25",
#                                     min_threshold=0.01,
#                                     max_threshold=0.25)
medical_infection_test = Observation(N = population,
                                     obs_frac = 1.,
                                     obs_status = 'I',
                                     obs_name = "0.01 < Infected(100%) < 1.",
                                     min_threshold=0.,
                                     max_threshold=1.0)

random_infection_test = Observation(N = population,
                                    obs_frac = 0.01,
                                    obs_status = 'I',
                                    obs_name = "Random Infection Test")

hospital_records = DataNodeObservation(N = population,
                                       bool_type=True,
                                       obs_status = 'H',
                                       obs_name = "Hospitalized from Data",
                                       specificity  = 0.999,
                                       sensitivity  = 0.999)

death_records = DataNodeObservation(N = population,
                                    bool_type=True,
                                    obs_status = 'D',
                                    obs_name = "Deceased from Data",
                                    specificity  = 0.999,
                                    sensitivity  = 0.999)

observations=[medical_infection_test,death_records,hospital_records]
plot_name_observations = "hrec_drec"

# give the data assimilator which transition rates and transmission rate to assimilate
transition_rates_to_update_str=['latent_periods', 'community_infection_periods', 'hospital_infection_periods']
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

# states_ensemble = deterministic_risk(contact_network,
#                               statuses,
#                               ensemble_size = ensemble_size)

epidemic_simulator.set_statuses(statuses)

fig, axes = plt.subplots(1, 3, figsize = (16, 4))

transition_rates_to_update_str=['latent_periods', 'community_infection_periods', 'hospital_infection_periods']

community_transmission_rate_trace = np.copy(community_transmission_rate_ensemble)
latent_periods_trace              = np.copy(np.array([member.latent_periods for member in transition_rates_ensemble]).reshape(-1,1))
community_infection_periods_trace = np.copy(np.array([member.community_infection_periods for member in transition_rates_ensemble]).reshape(-1,1))
hospital_infection_periods_trace  = np.copy(np.array([member.hospital_infection_periods for member in transition_rates_ensemble]).reshape(-1,1))

#contact_network_list = [copy.deepcopy(contact_network)]
#epidemic_simulator_states_list = [copy.deepcopy(statuses)]
#plot_epidemic_simulator_states_list = [copy.deepcopy(statuses)]
contact_network_list = []
epidemic_simulator_states_list = []
plot_epidemic_simulator_states_list = []

for i in range(int(simulation_length/static_contact_interval)):
    #Within the epidemic_simulator:
    # health_service discharge and admit patients [changes the contact network]
    # contact_simulator run [changes the mean contact duration on the given network]
    # set the new contact rates on the network! kinetic_model.set_mean_contact_duration(contact_duration)
    # run the kinetic model [kinetic produces the current statuses used as data]

    epidemic_simulator.run(stop_time = epidemic_simulator.time + static_contact_interval)
    contact_network_list.append(contact_network.copy())
    epidemic_simulator_states_list.append(copy.deepcopy(epidemic_simulator.kinetic_model.current_statuses))
    plot_epidemic_simulator_states_list.append(copy.deepcopy(epidemic_simulator.kinetic_model.statuses))

#total_steps = int(simulation_length/static_contact_interval)
#for i in range(1,total_steps):
#    stop_time = simulation_length - (i-1) * static_contact_interval
#    start_time = simulation_length - (i) * static_contact_interval
#    axes = plot_kinetic_model_data(epidemic_simulator.kinetic_model,
#                                   axes = axes,
#                                   statuses_input = True,
#                                   current_time = start_time,
#                                   statuses = plot_epidemic_simulator_states_list[total_steps-1-i])
#     
#    plt.savefig('da_ric_tprobs_'+plot_name_observations+'.png', rasterized=True, dpi=150)

#states_ensemble = deterministic_risk(contact_network,
#                              epidemic_simulator_states_list[-1],
#                              ensemble_size = ensemble_size)
states_ensemble = random_risk(contact_network,
                              fraction_infected = 0.01,
                              ensemble_size = ensemble_size)
master_eqn_ensemble.set_states_ensemble(states_ensemble)

total_steps = int(simulation_length/static_contact_interval)
for i in range(1,total_steps-1):

    # Backward DA
    stop_time = (i) * static_contact_interval
    start_time = (i-1) * static_contact_interval

    master_eqn_ensemble.set_mean_contact_duration() #do not need to reset weights as already set in kinetic model
    # would love to double check this! ^
    master_eqn_ensemble.contact_network = contact_network_list[i-1]
    states_ensemble_dict = master_eqn_ensemble.simulate_backwards(y0 = states_ensemble, \
                                                                  stop_time = stop_time, \
                                                                  start_time = start_time, \
                                                                  n_steps = 25)
    states_ensemble = np.copy(states_ensemble_dict['states'][:,:,-1])

    # perform data assimlation [update the master eqn states, the transition rates, and the transmission rate (if supplied)]
    (states_ensemble,
     transition_rates_ensemble,
     community_transmission_rate_ensemble
     ) = assimilator.update(ensemble_state = states_ensemble,
                            data = epidemic_simulator_states_list[i-1],
                            full_ensemble_transition_rates = transition_rates_ensemble,
                            full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                            user_network = contact_network_list[i-1])

    #update model parameters (transition and transmission rates) of the master eqn model
    master_eqn_ensemble.update_transition_rates(transition_rates_ensemble)
    master_eqn_ensemble.update_transmission_rate(community_transmission_rate_ensemble)

    master_eqn_ensemble.set_states_ensemble(states_ensemble)

    
    # Forward DA
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)

    # perform data assimlation [update the master eqn states, the transition rates, and the transmission rate (if supplied)]
    (states_ensemble,
     transition_rates_ensemble,
     community_transmission_rate_ensemble
     ) = assimilator.update(ensemble_state = states_ensemble,
                            data = epidemic_simulator_states_list[i],
                            full_ensemble_transition_rates = transition_rates_ensemble,
                            full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                            user_network = contact_network_list[i])

    #update model parameters (transition and transmission rates) of the master eqn model
    master_eqn_ensemble.update_transition_rates(transition_rates_ensemble)
    master_eqn_ensemble.update_transmission_rate(community_transmission_rate_ensemble)

    ## for tracking purposes
    #community_transmission_rate_trace = np.hstack([community_transmission_rate_trace, community_transmission_rate_ensemble])
    #latent_periods_trace              = np.hstack([latent_periods_trace, np.array([member.latent_periods for member in transition_rates_ensemble]).reshape(-1,1)])
    #community_infection_periods_trace = np.hstack([community_infection_periods_trace, np.array([member.community_infection_periods for member in transition_rates_ensemble]).reshape(-1,1)])
    #hospital_infection_periods_trace  = np.hstack([hospital_infection_periods_trace, np.array([member.hospital_infection_periods for member in transition_rates_ensemble]).reshape(-1,1)])

    #update states/statuses/times for next iteration
    master_eqn_ensemble.set_states_ensemble(states_ensemble)

    if i == 1:
        start_time = (i-1) * static_contact_interval
        stop_time = (i) * static_contact_interval
        axes = plot_ensemble_states(master_eqn_ensemble.states_trace,
                                    np.linspace(start_time, stop_time, 25 + 1),
                                    axes = axes,
                                    xlims = (-0.1, simulation_length),
                                    a_min = 0.0)

        axes = plot_kinetic_model_data(epidemic_simulator.kinetic_model,
                                       axes = axes,
                                       statuses_input = True,
                                       current_time = start_time,
                                       statuses = plot_epidemic_simulator_states_list[i])

    # Prediction
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)

    start_time = (i) * static_contact_interval
    stop_time = (i+1) * static_contact_interval
    axes = plot_ensemble_states(master_eqn_ensemble.states_trace,
                                np.linspace(start_time, stop_time, 25 + 1),
                                axes = axes,
                                xlims = (-0.1, simulation_length),
                                a_min = 0.0)

    axes = plot_kinetic_model_data(epidemic_simulator.kinetic_model,
                                   axes = axes,
                                   statuses_input = True,
                                   current_time = start_time,
                                   statuses = plot_epidemic_simulator_states_list[i+1])
     
    plt.savefig('forward_backward_da_ric_tprobs_'+plot_name_observations+'.png', rasterized=True, dpi=150)

#time_horizon = np.linspace(static_contact_interval, simulation_length, int(simulation_length/static_contact_interval))
#parameters = [community_transmission_rate_trace, latent_periods_trace, community_infection_periods_trace, hospital_infection_periods_trace ]
#parameters_names = ['transmission_rates', 'latent_periods', 'community_infection_periods', 'hospital_infection_periods']
#
#axes = plot_scalar_parameters(parameters, time_horizon, parameters_names)
#plt.savefig('backward_da_parameters_ric_tprobs_'+plot_name_observations +'.png', rasterized=True, dpi=150)
