#load the initialization for the epidemic etc.
from _epidemic_initializer import *

from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.user_base import FullUserGraphBuilder
from epiforecast.measurements import Observation, DataObservation, HighVarianceObservation
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.scenarios import random_epidemic
from epiforecast.risk_simulator_initial_conditions import deterministic_risk
from epiforecast.epiplots import plot_ensemble_states, plot_epidemic_data
from epiforecast.utilities import compartments_count

#
# create the  user_network (we do this here for plotting the epidemic)
#
user_network = network.build_user_network_using(FullUserGraphBuilder())

user_nodes = user_network.get_nodes()
user_population = user_network.get_node_count()

start_time = epidemic_simulator.time
simulation_length = 30
print("We first create an epidemic for",
      simulation_length,
      "days, then we solve the master equations forward for this time")

# set up the initial conditions
statuses = random_epidemic(population,
                           populace,
                           fraction_infected=0.01)

epidemic_simulator.set_statuses(statuses)

#for graphing against against users
user_statuses = { node : statuses[node] for node in user_nodes }
n_S, n_E, n_I, n_H, n_R, n_D = compartments_count(user_statuses)
statuses_sum_trace = [[n_S, n_E, n_I, n_H, n_R, n_D]]

time = start_time
time_trace = np.arange(time,simulation_length,static_contact_interval)

# Run the epidemic simulation and store the results
for i in range(int(simulation_length/static_contact_interval)):
    network = epidemic_simulator.run(stop_time = epidemic_simulator.time + static_contact_interval,
                                     current_network = network)

    #save the start time network and statuses
    epidemic_data_storage.save_network_by_start_time(contact_network=network, start_time=time)
    epidemic_data_storage.save_start_statuses_to_network(start_time=time, start_statuses=statuses)

    #update the statuses and time
    statuses = epidemic_simulator.kinetic_model.current_statuses
    time=epidemic_simulator.time

    #save the statuses at the new time
    epidemic_data_storage.save_end_statuses_to_network(end_time=time, end_statuses=statuses)

    #statuses of user base
    user_statuses = { node : statuses[node] for node in user_nodes }
    n_S, n_E, n_I, n_H, n_R, n_D = compartments_count(user_statuses)
    statuses_sum_trace.append([n_S, n_E, n_I, n_H, n_R, n_D])

fig, axes = plt.subplots(1, 3, figsize = (16, 4))
axes = plot_epidemic_data(population = population,
                       statuses_list = statuses_sum_trace,
                                axes = axes,
                          plot_times = time_trace)

plt.savefig('backward_filter_on_loaded_epidemic.png', rasterized=True, dpi=150)


#
# Reset the world-time to 0, load the initial network
#

time = 0.0 
loaded_data = epidemic_data_storage.get_network_from_start_time(start_time=time)
user_network = loaded_data.contact_network.build_user_network_using(FullUserGraphBuilder())
initial_statuses = loaded_data.start_statuses

#
# Set up the population priors
#

ensemble_size = 100 

transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(transition_rates)

#set transmission_rates
community_transmission_rate_ensemble = community_transmission_rate * np.ones([ensemble_size,1])

transition_rates_to_update_str = []
transmission_rate_to_update_flag = False 

#
# Set up the data assimilator
#

# imperfect observations
random_infection_test = Observation(N = user_population,
                             obs_frac = 1.0,
                           obs_status = 'I',
                             obs_name = "Random Infection Test",
                          obs_var_min = 1e-6)

high_var_infection_test = HighVarianceObservation(N = user_population,
                                           obs_frac = 0.1,
                                         obs_status = 'I',
                                           obs_name = "Test maximal variance infected",
                                        obs_var_min = 1e-6)

# perfect observations
positive_hospital_records = DataObservation(N = user_population,
                                       set_to_one=True,
                                       obs_status = 'H',
                                       obs_name = "hospstate")

negative_hospital_records = DataObservation(N = user_population,
                                    set_to_one=False,
                                    obs_status = 'H',
                                    obs_name = "nohospstate")

positive_death_records = DataObservation(N = user_population,
                                    set_to_one=True,
                                    obs_status = 'D',
                                    obs_name = "deathstate")

negative_death_records = DataObservation(N = user_population,
                                    set_to_one=False,
                                    obs_status = 'D',
                                    obs_name = "nodeathstate")

imperfect_observations=[random_infection_test]

perfect_observations=[positive_hospital_records,
                      negative_hospital_records,
                      positive_death_records,
                      negative_death_records]

# create the assimilator
assimilator_imperfect_observations = DataAssimilator(observations = imperfect_observations,
                                                           errors = [],
                                   transition_rates_to_update_str = transition_rates_to_update_str,
                                 transmission_rate_to_update_flag = transmission_rate_to_update_flag)

assimilator_perfect_observations = DataAssimilator(observations = perfect_observations,
                                                         errors = [],
                                 transition_rates_to_update_str = transition_rates_to_update_str,
                               transmission_rate_to_update_flag = transmission_rate_to_update_flag)

#
# Set up the ensemble of master equtions
#

master_eqn_ensemble = MasterEquationModelEnsemble(population = user_population,
                                                 transition_rates = transition_rates_ensemble,
                                                transmission_rate = community_transmission_rate_ensemble,
                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                    ensemble_size = ensemble_size,
                                                    start_time = start_time)

#
# Run the master equations on the loaded networks
#

states_trace_ensemble=np.zeros([ensemble_size,5*user_population,time_trace.size])

states_ensemble = deterministic_risk(user_nodes,
                                     initial_statuses,
                                     ensemble_size = ensemble_size)

master_eqn_ensemble.set_states_ensemble(states_ensemble)

forward_run_time = time
master_eqn_ensemble.set_start_time(time)
for j in range(int(simulation_length/static_contact_interval)):

    loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=forward_run_time)
    user_network = loaded_data.contact_network.build_user_network_using(FullUserGraphBuilder())
    user_nodes = user_network.get_nodes()
    master_eqn_ensemble.set_mean_contact_duration(loaded_data.contact_network.get_edge_weights())
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)

    forward_run_time = forward_run_time + static_contact_interval
    master_eqn_ensemble.set_states_ensemble(states_ensemble)
    master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
                                       new_transmission_rate = community_transmission_rate_ensemble)

#
# Run backward DA
#

time = simulation_length

master_eqn_ensemble.set_states_ensemble(states_ensemble)

states_trace_ensemble[:,:,0] = states_ensemble

master_eqn_ensemble.set_start_time(time)
for i in range(int(simulation_length/static_contact_interval)):

    loaded_data=epidemic_data_storage.get_network_from_end_time(end_time=time)
    user_network = loaded_data.contact_network.build_user_network_using(FullUserGraphBuilder())
    user_nodes = user_network.get_nodes()
    master_eqn_ensemble.set_mean_contact_duration(loaded_data.contact_network.get_edge_weights())
    states_ensemble = master_eqn_ensemble.simulate_backwards(static_contact_interval, n_steps = 25)

    time = time - static_contact_interval
    
    (states_ensemble,
     transition_rates_ensemble,
     community_transmission_rate_ensemble
    ) = assimilator_imperfect_observations.update(ensemble_state = states_ensemble,
                                                            data = loaded_data.start_statuses,
                                  full_ensemble_transition_rates = transition_rates_ensemble,
                                 full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                                      user_nodes = user_nodes,
                                                            time = time)

    (states_ensemble,
     transition_rates_ensemble,
     community_transmission_rate_ensemble
    ) = assimilator_perfect_observations.update(ensemble_state = states_ensemble,
                                                          data = loaded_data.start_statuses,
                                full_ensemble_transition_rates = transition_rates_ensemble,
                               full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                                    user_nodes = user_nodes,
                                                          time = time)

    #update model parameters (transition and transmission rates) of the master eqn model

    #at the update the time
    
    master_eqn_ensemble.set_states_ensemble(states_ensemble)
    master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
                                       new_transmission_rate = community_transmission_rate_ensemble)

    states_trace_ensemble[:,:,i] = states_ensemble

axes = plot_ensemble_states(population,
                            states_trace_ensemble,
                            np.flip(time_trace),
                            axes = axes,
                            xlims = (-0.1, simulation_length),
                            a_min = 0.0)

plt.savefig('backward_filter_on_loaded_epidemic.png', rasterized=True, dpi=150)
