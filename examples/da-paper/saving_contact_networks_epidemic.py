import os, sys; sys.path.append(os.path.join("../.."))

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
from epiforecast.risk_simulator_initial_conditions import deterministic_risk


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
simulation_length = 2

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

# Create storage for networks and data
epidemic_data_storage = StaticIntervalDataSeries(static_contact_interval)

# set up the initial conditions
fraction_infected=0.01
statuses = random_epidemic(contact_network,
                           fraction_infected=0.01)
epidemic_simulator.set_statuses(statuses)
# for use later initializing the master equations
initial_statuses=copy.deepcopy(statuses) 

time=0.0
time_trace = np.arange(time,simulation_length,static_contact_interval)
statuses_sum_trace = [[population-int(fraction_infected*population), 0, int(fraction_infected*population),0,0,0]]


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


    statuses_sum_trace.append([epidemic_simulator.kinetic_model.statuses['S'][-1],
                           epidemic_simulator.kinetic_model.statuses['E'][-1],
                           epidemic_simulator.kinetic_model.statuses['I'][-1],
                           epidemic_simulator.kinetic_model.statuses['H'][-1],
                           epidemic_simulator.kinetic_model.statuses['R'][-1],
                           epidemic_simulator.kinetic_model.statuses['D'][-1]]) 

axes = plot_epidemic_data(kinetic_model = epidemic_simulator.kinetic_model,
                          statuses_list = statuses_sum_trace,
                                   axes = axes,
                             plot_times = time_trace)

plt.savefig('save_and_load_epidemic.png', rasterized=True, dpi=150)


#
# Set up the ensemble of master equtions
#

ensemble_size = 1 

transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(transition_rates)

#set transmission_rates
community_transmission_rate_ensemble = community_transmission_rate*np.ones([ensemble_size,1]) 

master_eqn_ensemble = MasterEquationModelEnsemble(contact_network = contact_network,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = community_transmission_rate_ensemble,
                                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                  ensemble_size = ensemble_size)


#
# Run the master equations on the loaded networks
#

time = 0.0

states_ensemble = deterministic_risk(contact_network,
                                     initial_statuses,
                                     ensemble_size = ensemble_size)

master_eqn_ensemble.set_states_ensemble(states_ensemble)

states_trace_ensemble=np.zeros([ensemble_size,5*population,time_trace.size])
states_trace_ensemble[:,:,0] = states_ensemble

for i in range(int(simulation_length/static_contact_interval)):

    loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=time)
    master_eqn_ensemble.set_contact_network_and_contact_duration(loaded_data.contact_network) # contact duration stored on network
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)
    
    #at the update the time
    time = time + static_contact_interval
    master_eqn_ensemble.set_states_ensemble(states_ensemble)

    states_trace_ensemble[:,:,i] = states_ensemble

axes = plot_ensemble_states(states_trace_ensemble,
                            time_trace,
                            axes = axes,
                            xlims = (-0.1, simulation_length),
                            a_min = 0.0)
    
plt.savefig('save_and_load_epidemic.png', rasterized=True, dpi=150)

