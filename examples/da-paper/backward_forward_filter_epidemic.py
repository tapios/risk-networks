import os, sys; sys.path.append(os.path.join("../.."))
import pdb

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

from numba import set_num_threads

set_num_threads(1)

from epiforecast.populations import assign_ages,  TransitionRates
from epiforecast.samplers import AgeDependentConstant

from epiforecast.scenarios import load_edges, random_epidemic
from epiforecast.epiplots import plot_ensemble_states, plot_epidemic_data
from epiforecast.node_identifier_helper import load_node_identifiers
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.utilities import seed_three_random_states
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
from epiforecast.user_base import FullUserBase
from epiforecast.measurements import Observation, DataObservation
from epiforecast.data_assimilator import DataAssimilator




#def deterministic_risk(contact_network, initial_statuses, ensemble_size=1):
#
#    population = len(contact_network)
#    states_ensemble = np.zeros([ensemble_size, 5 * population])
#
#    init_catalog = {'S': False, 'I': True}
#    infected = np.array([init_catalog[status] for status in list(initial_statuses.values())])
#
#    for mm in range(ensemble_size):
#        E, I, H, R, D = np.zeros([5, population])
#        S = np.ones(population,)
#        I[infected] = 1.
#        S[infected] = 0.
#
#        states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))
#
#    return states_ensemble

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

seed = 942395
seed_three_random_states(seed)


#
# Load an example network
#




edges = load_edges(os.path.join('../..', 'data', 'networks', 'edge_list_SBM_1e3_nobeds.txt'))
node_identifiers = load_node_identifiers(os.path.join('../..', 'data', 'networks', 'node_identifier_SBM_1e3_nobeds.txt'))

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)

#
# create the (Full) user_base
#

user_base = FullUserBase(contact_network)
users = list(user_base.contact_network.nodes)
user_population=len(user_base.contact_network)

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
# Set up the epidemic simulator and health service
#

minute = 1 / 60 / 24
hour = 60 * minute

mean_contact_lifetime=0.5*minute
static_contact_interval = 3 * hour
simulation_length = 30 

health_service = HealthService(static_population_network = contact_network,
                               health_workers = node_identifiers['health_workers'])

epidemic_simulator = EpidemicSimulator(
                 contact_network = contact_network,
                transition_rates = transition_rates,
     community_transmission_rate = community_transmission_rate,
 hospital_transmission_reduction = hospital_transmission_reduction,
         static_contact_interval = static_contact_interval,
           mean_contact_lifetime = mean_contact_lifetime,
              day_inception_rate = 22,
            night_inception_rate = 2,
                  health_service = health_service)



#
# Run the epidemic simulation and store the results
#
time=0.0

# Create storage for networks and data
epidemic_data_storage = StaticIntervalDataSeries(static_contact_interval)

# set up the initial conditions
fraction_infected=0.01
statuses = random_epidemic(contact_network,
                           fraction_infected=0.01)
epidemic_simulator.set_statuses(statuses)
# for use later initializing the master equations

#for graphing against against users
Scount=len([node for node in users if statuses[node] == 'S'])
Ecount=len([node for node in users if statuses[node] == 'E'])
Icount=len([node for node in users if statuses[node] == 'I'])
Hcount=len([node for node in users if statuses[node] == 'H'])
Rcount=len([node for node in users if statuses[node] == 'R'])
Dcount=len([node for node in users if statuses[node] == 'D'])

time_trace = np.arange(time,simulation_length,static_contact_interval)
statuses_sum_trace = [[Scount,Ecount,Icount,Hcount,Rcount,Dcount]]


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

    #statuses of user base (here Full)
    Scount=len([node for node in users if statuses[node] == 'S'])
    Ecount=len([node for node in users if statuses[node] == 'E'])
    Icount=len([node for node in users if statuses[node] == 'I'])
    Hcount=len([node for node in users if statuses[node] == 'H'])
    Rcount=len([node for node in users if statuses[node] == 'R'])
    Dcount=len([node for node in users if statuses[node] == 'D'])

    statuses_sum_trace.append([Scount,Ecount,Icount,Hcount,Rcount,Dcount])

axes = plot_epidemic_data(kinetic_model = epidemic_simulator.kinetic_model,
                          statuses_list = statuses_sum_trace,
                                   axes = axes,
                             plot_times = time_trace)

plt.savefig('backward_forward_filter_on_loaded_epidemic.png', rasterized=True, dpi=150)


#
# Reset the world-time to 0, load the initial network
#

backward_DA_interval = 1
forward_DA_interval = 2

time = backward_DA_interval 
loaded_data = epidemic_data_storage.get_network_from_end_time(end_time=time)
user_network = loaded_data.contact_network.subgraph(users)
initial_statuses = loaded_data.start_statuses

#
# Set up the population priors
#

ensemble_size = 100 

transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(
        TransitionRates(user_network,
                        latent_periods = 3.7,
           community_infection_periods = 3.2,
            hospital_infection_periods = 5.0,
              hospitalization_fraction = AgeDependentConstant([0.002,  0.01,   0.04, 0.076,  0.16]),
          community_mortality_fraction = AgeDependentConstant([ 1e-4,  1e-3,  0.001,  0.07, 0.015]),
           hospital_mortality_fraction = AgeDependentConstant([0.019, 0.073,  0.193, 0.327, 0.512]))
                                     )
 
#set transmission_rates
community_transmission_rate_ensemble = community_transmission_rate * np.ones([ensemble_size,1])

transition_rates_to_update_str = []
transmission_rate_to_update_flag = False 

#
# Set up the data assimilator
#

random_infection_test = Observation(N = user_population,
                             obs_frac = 1.0,
                           obs_status = 'I',
                             obs_name = "Random Infection Test",
                          obs_var_min = 1e-6)

observations=[random_infection_test]

# create the assimilator
assimilator = DataAssimilator(observations = observations,
                                    errors = [],
            transition_rates_to_update_str = transition_rates_to_update_str,
          transmission_rate_to_update_flag = transmission_rate_to_update_flag)

#
# Set up the ensemble of master equtions
#

master_eqn_ensemble = MasterEquationModelEnsemble(contact_network = user_network,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = community_transmission_rate_ensemble,
                                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                  ensemble_size = ensemble_size)

#
# Run the master equations on the loaded networks
#

time = backward_DA_interval 

states_ensemble = deterministic_risk(user_network,
                                     initial_statuses,
                                     ensemble_size = ensemble_size)

master_eqn_ensemble.set_states_ensemble(states_ensemble)

states_trace_ensemble=np.zeros([ensemble_size,5*population,time_trace.size])
states_trace_ensemble[:,:,0] = states_ensemble

for k in range(1,int(simulation_length/backward_DA_interval)):
    backward_DA_time = k*backward_DA_interval
    master_eqn_ensemble.set_start_time(backward_DA_time)
    for i in range(int(backward_DA_interval/static_contact_interval)):
        loaded_data=epidemic_data_storage.get_network_from_end_time(end_time=backward_DA_time)
        user_network = loaded_data.contact_network.subgraph(users)
        master_eqn_ensemble.set_contact_network_and_contact_duration(user_network) # contact duration stored on network
        states_ensemble = master_eqn_ensemble.simulate_backwards(static_contact_interval, n_steps = 25)
        
    
        (states_ensemble,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
        ) = assimilator.update(ensemble_state = states_ensemble,
                                         data = loaded_data.start_statuses,
               full_ensemble_transition_rates = transition_rates_ensemble,
              full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                 user_network = user_network)
        
        #update model parameters (transition and transmission rates) of the master eqn model
        
        #at the update the time
        backward_DA_time = backward_DA_time - static_contact_interval
        master_eqn_ensemble.set_states_ensemble(states_ensemble)
        master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
                                           new_transmission_rate = community_transmission_rate_ensemble)

    forward_DA_time = backward_DA_time
    master_eqn_ensemble.set_start_time(forward_DA_time)
    for j in range(int(backward_DA_interval/static_contact_interval)):
    
        loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=forward_DA_time)
        user_network = loaded_data.contact_network.subgraph(users)
        master_eqn_ensemble.set_contact_network_and_contact_duration(user_network) # contact duration stored on network
        states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)
    
    
        (states_ensemble,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
        ) = assimilator.update(ensemble_state = states_ensemble,
                                         data = loaded_data.end_statuses,
               full_ensemble_transition_rates = transition_rates_ensemble,
              full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                 user_network = user_network)
    
        #update model parameters (transition and transmission rates) of the master eqn model
    
        #at the update the time
        forward_DA_time = forward_DA_time + static_contact_interval
        master_eqn_ensemble.set_states_ensemble(states_ensemble)
        master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
                                           new_transmission_rate = community_transmission_rate_ensemble)
        if k == 1:
            states_trace_ensemble[:,:,j] = states_ensemble

    forward_prediction_time = forward_DA_time
    master_eqn_ensemble.set_start_time(forward_prediction_time)
    for j in range(int((forward_DA_interval-backward_DA_interval)/static_contact_interval)):
    
        loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=forward_prediction_time)
        user_network = loaded_data.contact_network.subgraph(users)
        master_eqn_ensemble.set_contact_network_and_contact_duration(user_network) # contact duration stored on network
        states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)
    
        #(states_ensemble,
        # transition_rates_ensemble,
        # community_transmission_rate_ensemble
        #) = assimilator.update(ensemble_state = states_ensemble,
        #                                 data = loaded_data.end_statuses,
        #       full_ensemble_transition_rates = transition_rates_ensemble,
        #      full_ensemble_transmission_rate = community_transmission_rate_ensemble,
        #                         user_network = user_network)
    
        #update model parameters (transition and transmission rates) of the master eqn model
    
        #at the update the time
        forward_prediction_time = forward_prediction_time + static_contact_interval
        master_eqn_ensemble.set_states_ensemble(states_ensemble)
        #master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
        #                                   new_transmission_rate = community_transmission_rate_ensemble)

        states_trace_ensemble[:,:,int(k*backward_DA_interval/static_contact_interval)+j] = states_ensemble

axes = plot_ensemble_states(states_trace_ensemble,
                            time_trace,
                            axes = axes,
                            xlims = (-0.1, simulation_length),
                            a_min = 0.0)
    
plt.savefig('backward_forward_filter_on_loaded_epidemic.png', rasterized=True, dpi=150)

