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

from epiforecast.populations import assign_ages, TransitionRates
from epiforecast.samplers import AgeDependentConstant

from epiforecast.scenarios import load_edges, random_epidemic

from epiforecast.epiplots import plot_ensemble_states, plot_epidemic_data
from epiforecast.node_identifier_helper import load_node_identifiers
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.utilities import seed_numba_random_state
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
from epiforecast.user_base import FullUserBase, ContiguousUserBase, assign_user_connectivity_to_contact_network
from epiforecast.measurements import Observation
from epiforecast.data_assimilator import DataAssimilator

def deterministic_risk(
        contact_network,
        initial_states,
        ensemble_size=1):

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

def global_risk(
        contact_network,
        fraction_infected,
        ensemble_size=1):

    population = len(contact_network)
    states_ensemble = np.zeros([ensemble_size, 5 * population])

    for mm in range(ensemble_size):
        E, I, H, R, D = np.zeros([5, population])
        S = (1-fraction_infected)*np.ones(population,)
        I = fraction_infected*np.ones(population,)
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
   hospital_mortality_fraction = AgeDependentConstant([0.019, 0.073,  0.193, 0.327, 0.512]))

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

epidemic_simulator = EpidemicSimulator(contact_network = contact_network,
                                      transition_rates = transition_rates,
                           community_transmission_rate = community_transmission_rate,
                       hospital_transmission_reduction = hospital_transmission_reduction,
                               static_contact_interval = static_contact_interval,
                                 mean_contact_lifetime = mean_contact_lifetime,
                                    day_inception_rate = 22,
                                  night_inception_rate = 2,
                                        health_service = health_service,
                                            start_time = start_time)

ensemble_size = 100


#user_base = FullUserBase(contact_network)
user_fraction=0.5
user_base = ContiguousUserBase(
    contact_network,
    user_fraction,
    method="neighbor",
    seed_user=None)

users = list(user_base.contact_network.nodes)
user_population=len(user_base.contact_network)

# we assign a score [0, 1] of how many inner and outer connections the users have.
assign_user_connectivity_to_contact_network(contact_network, user_base.contact_network)

transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(
        TransitionRates(user_base.contact_network,
                        latent_periods = 3.7,
           community_infection_periods = 3.2,
            hospital_infection_periods = 5.0,
              hospitalization_fraction = AgeDependentConstant([0.002,  0.01,   0.04, 0.076,  0.16]),
          community_mortality_fraction = AgeDependentConstant([ 1e-4,  1e-3,  0.001,  0.07, 0.015]),
           hospital_mortality_fraction = AgeDependentConstant([0.019, 0.073,  0.193, 0.327, 0.512]))
                                     )
 
#set transmission_rates
community_transmission_rate_ensemble = np.random.normal(0, 0.2, size=[ensemble_size,1])
#np.log(community_transmission_rate * np.ones([ensemble_size,1]) )
exogenous_transmission_rate_ensemble = np.random.normal(0, 0.2, size=[ensemble_size,1])

master_eqn_ensemble = MasterEquationModelEnsemble(contact_network = user_base.contact_network,
                                                 transition_rates = transition_rates_ensemble,
                                                transmission_rate = np.exp(community_transmission_rate_ensemble),
                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                      exogenous_transmission_rate = np.exp(exogenous_transmission_rate_ensemble),
                                                    ensemble_size = ensemble_size,
                                                       start_time = start_time)


#Data assimilator: Observations:
random_infection_test = Observation(user_population = user_population,
                                           obs_frac = 1.0,
                                         obs_status = 'I',
                                           obs_name = "Random Infection Test",
                                    min_threshold = 0.0,
                                    specificity = 0.999,
                                    sensitivity = 0.999)

observations=[random_infection_test]

plot_name_observations = "randitest"

# give the data assimilator which transition rates and transmission rates to assimilate
transition_rates_to_update_str = []
transmission_rate_to_update_flag = True
exogenous_transmission_rate_to_update_flag = True

# create the assimilator
assimilator = DataAssimilator(observations = observations,
                                    errors = [],
            transition_rates_to_update_str = transition_rates_to_update_str,
          transmission_rate_to_update_flag = transmission_rate_to_update_flag,
exogenous_transmission_rate_to_update_flag = exogenous_transmission_rate_to_update_flag)


# Create storage for networks and data
epidemic_data_storage = StaticIntervalDataSeries(static_contact_interval)

time = start_time

statuses = random_epidemic(contact_network, fraction_infected=0.01)
user_statuses = {node: statuses[node] for node in user_base.contact_network.nodes}
states_ensemble = deterministic_risk(
    user_base.contact_network,
    user_statuses,
    ensemble_size = ensemble_size)

epidemic_simulator.set_statuses(statuses)
master_eqn_ensemble.set_states_ensemble(states_ensemble)

time_trace = np.linspace(start=start_time, stop=simulation_length,
                         num=int(simulation_length/static_contact_interval)+1)

Scount=len([node for node in users if statuses[node] == 'S'])
Ecount=len([node for node in users if statuses[node] == 'E'])
Icount=len([node for node in users if statuses[node] == 'I'])
Hcount=len([node for node in users if statuses[node] == 'H'])
Rcount=len([node for node in users if statuses[node] == 'R'])
Dcount=len([node for node in users if statuses[node] == 'D'])

statuses_sum_trace=[[Scount,Ecount,Icount,Hcount,Rcount,Dcount]]
states_trace_ensemble = np.zeros([ensemble_size,5*user_population,time_trace.size])
states_trace_ensemble[:,:,0] = states_ensemble


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

    #user_statuses = {node : statuses[node] for node in users}
    Scount=len([node for node in users if statuses[node] == 'S'])
    Ecount=len([node for node in users if statuses[node] == 'E'])
    Icount=len([node for node in users if statuses[node] == 'I'])
    Hcount=len([node for node in users if statuses[node] == 'H'])
    Rcount=len([node for node in users if statuses[node] == 'R'])
    Dcount=len([node for node in users if statuses[node] == 'D'])

    statuses_sum_trace.append([Scount,Ecount,Icount,Hcount,Rcount,Dcount])


axes = plot_epidemic_data(
    kinetic_model = epidemic_simulator.kinetic_model,
    statuses_list = statuses_sum_trace,
    axes = axes,
    plot_times = time_trace)

plt.savefig('da_with_'+str(user_fraction)+'_user_base_'+plot_name_observations+'.png', rasterized=True, dpi=150)


time = start_time

# Then we run the master equations and data assimilator forward on the loaded networks
for i in range(int(simulation_length/static_contact_interval)):

    loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=time)
    user_network = loaded_data.contact_network.subgraph(users)
    master_eqn_ensemble.set_contact_network_and_contact_duration(user_network) # contact duration stored on network
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)
    

    (states_ensemble,
     transition_rates_ensemble,
     community_transmission_rate_ensemble,
     exogenous_transmission_rate_ensemble
    ) = assimilator.update(ensemble_state = states_ensemble,
                                     data = loaded_data.end_statuses,
           full_ensemble_transition_rates = transition_rates_ensemble,
          full_ensemble_transmission_rate = community_transmission_rate_ensemble,
full_ensemble_exogenous_transmission_rate = exogenous_transmission_rate_ensemble,
                             user_network = user_network)
    
    #update model parameters (transition and transmission rates) of the master eqn model
    # adding some noise if necessary
    if transmission_rate_to_update_flag:
        community_transmission_rate_ensemble += np.random.normal(0,0.1,size=[ensemble_size,1])
    if exogenous_transmission_rate_to_update_flag:
        exogenous_transmission_rate_ensemble += np.random.normal(0,0.1,size=[ensemble_size,1])

    print("mean of exp(users) ", np.mean(np.exp(community_transmission_rate_ensemble)))
    print("var of exp(users) ",    np.var(np.exp(community_transmission_rate_ensemble)))
    print("mean of exp(exogenous) ", np.mean(np.exp(exogenous_transmission_rate_ensemble)))
    print("var of exp(exogenous) ",    np.var(np.exp(exogenous_transmission_rate_ensemble)))
    
    master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
                                       new_transmission_rate = np.exp(community_transmission_rate_ensemble),
                             new_exogenous_transmission_rate = np.exp(exogenous_transmission_rate_ensemble))
    
   

    #at the update the time
    time = time + static_contact_interval
    master_eqn_ensemble.set_states_ensemble(states_ensemble)

    states_trace_ensemble[:,:,i+1] = states_ensemble

axes = plot_ensemble_states(
    states_trace_ensemble,
    time_trace,
    axes = axes,
    xlims = (-0.1, simulation_length),
    a_min = 0.0)
    
plt.savefig('da_with_'+str(user_fraction)+'_user_base_'+plot_name_observations+'.png', rasterized=True, dpi=150)


