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

from epiforecast.populations import TransitionRates
from epiforecast.samplers import AgeDependentConstant

from epiforecast.scenarios import random_epidemic

from epiforecast.epiplots import plot_ensemble_states, plot_kinetic_model_data, plot_scalar_parameters, plot_epidemic_data
from epiforecast.contact_network import ContactNetwork
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.utilities import seed_numba_random_state
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
from epiforecast.user_base import FullUserBase, ContiguousUserBase, assign_user_connectivity_to_contact_network
from epiforecast.risk_simulator_initial_conditions import deterministic_risk

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

edges_filename = os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3_nobeds.txt')
identifiers_filename = os.path.join('..', 'data', 'networks', 'node_identifier_SBM_1e3_nobeds.txt')

network = ContactNetwork.from_files(edges_filename,identifiers_filename)
population = network.get_node_count()
populace = network.get_nodes()


start_time = 0.0

minute = 1 / 60 / 24
hour = 60 * minute
simulation_length = 30

#
# Clinical parameters of an age-distributed population
#

age_distribution =[0.21, 0.4, 0.25, 0.08, 0.06]
network.draw_and_set_age_groups(age_distribution)

# We process the clinical data to determine transition rates between each epidemiological state,
latent_periods = 3.7
community_infection_periods = 3.2
hospital_infection_periods = 5.0
hospitalization_fraction = AgeDependentConstant([0.002,  0.01,   0.04, 0.076,  0.16])
community_mortality_fraction = AgeDependentConstant([ 1e-4,  1e-3,  0.001,  0.07,  0.015])
hospital_mortality_fraction = AgeDependentConstant([0.019, 0.073,  0.193, 0.327, 0.512])

transition_rates = TransitionRates(population = network.get_node_count(),
                                   lp_sampler = latent_periods,
                                  cip_sampler = community_infection_periods,
                                  hip_sampler = hospital_infection_periods,
                                   hf_sampler = hospitalization_fraction,
                                  cmf_sampler = community_mortality_fraction,
                                  hmf_sampler = hospital_mortality_fraction,
                    distributional_parameters = network.get_age_groups()
)

transition_rates.calculate_from_clinical() 

network.set_transition_rates_for_kinetic_model(transition_rates)

community_transmission_rate = 12.0

#
# Create the user base
#

# for plotting purposes we need to know the users before the master equations are run
# So we initialize the user_network here.
#user_network = network.build_user_network_using(FullUserGraphBuilder())
user_fraction=0.5
network.build_user_network_using(
        ContiguousUserGraphBuilder(user_fraction,
                                   method="neighbor",
                                   seed_user=None))

user_nodes = list(user_base.contact_network.get_nodes())
user_population=user_network.get_node_count()
user_connectivity_score = calc_user_connectivity_score(contact_network, user_base.contact_network)

print("We first create an epidemic for",simulation_length,"days, then we solve the master equations with a user base of size",user_population,"forward for this time")



#
# Setup the the epidemic simulator
#

static_contact_interval = 3 * hour


health_service = HealthService(original_contact_network = network,
                               health_workers = network.get_health_workers())

mean_contact_lifetime=0.5*minute
hospital_transmission_reduction = 0.1
min_inception_rate = 2
max_inception_rate = 22

epidemic_simulator = EpidemicSimulator(
                 contact_network =network,
     community_transmission_rate = community_transmission_rate,
 hospital_transmission_reduction = hospital_transmission_reduction,
         static_contact_interval = static_contact_interval,
           mean_contact_lifetime = mean_contact_lifetime,
              day_inception_rate = max_inception_rate,
            night_inception_rate = min_inception_rate,
                  health_service = health_service,
                      start_time = start_time
                                      )

# Create storage for networks and data
epidemic_data_storage = StaticIntervalDataSeries(static_contact_interval)

time = start_time

statuses = random_epidemic(contact_network,
                           fraction_infected=0.01)

epidemic_simulator.set_statuses(statuses)


#
# First we run and save the epidemic
#

time_trace = np.arange(start_time,
                       simulation_length,
                       static_contact_interval)
statuses_sum_trace=[[population-int(0.01*population), 0, int(0.01*population),0,0,0]]



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
    Scount=len([node for node in user_nodes if statuses[node] == 'S'])
    Ecount=len([node for node in user_nodes if statuses[node] == 'E'])
    Icount=len([node for node in user_nodes if statuses[node] == 'I'])
    Hcount=len([node for node in user_nodes if statuses[node] == 'H'])
    Rcount=len([node for node in user_nodes if statuses[node] == 'R'])
    Dcount=len([node for node in user_nodes if statuses[node] == 'D'])

    statuses_sum_trace.append([Scount,Ecount,Icount,Hcount,Rcount,Dcount])


axes = plot_epidemic_data(kinetic_model = epidemic_simulator.kinetic_model,
                          statuses_list = statuses_sum_trace,
                                   axes = axes,
                             plot_times = time_trace)

plt.savefig('kinetic_and_master_with_user_base.png', rasterized=True, dpi=150)


#
# Reset time to start time
#
time = start_time


ensemble_size = 1

transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(
        TransitionRates(usert_network,
                        latent_periods = 3.7,
           community_infection_periods = 3.2,
            hospital_infection_periods = 5.0,
              hospitalization_fraction = AgeDependentConstant([0.002,  0.01,   0.04, 0.076,  0.16]),
          community_mortality_fraction = AgeDependentConstant([ 1e-4,  1e-3,  0.001,  0.07, 0.015]),
           hospital_mortality_fraction = AgeDependentConstant([0.019, 0.073,  0.193, 0.327, 0.512]))
                                     )
 
#set transmission_rates
community_transmission_rate_ensemble = 1.0*community_transmission_rate * np.ones([ensemble_size,1]) 
exogenous_transmission_rate_ensemble = 0.002*community_transmission_rate * np.ones([ensemble_size,1])

master_eqn_ensemble = MasterEquationModelEnsemble(contact_network = user_base.contact_network,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = community_transmission_rate_ensemble,
                                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                  exogenous_transmission_rate = exogenous_transmission_rate_ensemble,
                                                  user_connectivity_score = user_connectivity_score
                                                  ensemble_size = ensemble_size,
                                                  start_time = start_time)




states_ensemble = global_risk(user_base.contact_network,
                              fraction_infected = 0.01,
                              ensemble_size = ensemble_size)
master_eqn_ensemble.set_states_ensemble(states_ensemble)

states_trace_ensemble = np.zeros([ensemble_size,5*user_population,time_trace.size])
states_trace_ensemble[:,:,0] = states_ensemble

# Then we run the master equations forward on the loaded networks
for i in range(int(simulation_length/static_contact_interval)):

    loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=time)
    user_network = loaded_data.contact_network.subgraph(user_nodes)
    master_eqn_ensemble.set_contact_network_and_contact_duration(user_network) # contact duration stored on network
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)
    
    #at the update the time
    time = time + static_contact_interval
    master_eqn_ensemble.set_states_ensemble(states_ensemble)

    states_trace_ensemble[:,:,i+1] = states_ensemble

axes = plot_ensemble_states(states_trace_ensemble,
                            time_trace,
                            axes = axes,
                            xlims = (-0.1, simulation_length),
                            a_min = 0.0)
    
plt.savefig('kinetic_and_master_with_user_base.png', rasterized=True, dpi=150)


