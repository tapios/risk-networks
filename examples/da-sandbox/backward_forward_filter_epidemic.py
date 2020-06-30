#load the initialization for the epidemic etc.
from _epidemic_initializer import *

from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
from epiforecast.user_base import FullUserGraphBuilder, ContiguousUserGraphBuilder
from epiforecast.measurements import Observation, DataObservation, HighVarianceObservation
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
from epiforecast.scenarios import random_epidemic
from epiforecast.risk_simulator_initial_conditions import deterministic_risk, uniform_risk, random_risk
from epiforecast.epiplots import plot_ensemble_states, plot_epidemic_data

#
# create the  user_network (we do this here for plotting the epidemic)
#
user_network = network.build_user_network_using(FullUserGraphBuilder())
#user_fraction = 0.1
#user_network= network.build_user_network_using(FractionalUserGraphBuilder(user_fraction))

user_nodes = user_network.get_nodes()
user_population=user_network.get_node_count()


def deterministic_risk(population, initial_states, ensemble_size=1):

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
# Run the epidemic simulation and store the results
#
start_time = epidemic_simulator.time
time = start_time
simulation_length = 30 
print("We first create an epidemic for",
      simulation_length,
      "days, then we solve the master equations forward for this time")

# Create storage for networks and data
epidemic_data_storage = StaticIntervalDataSeries(static_contact_interval)

# set up the initial conditions
fraction_infected=0.01
statuses = random_epidemic(population,
                           populace,
                           fraction_infected=0.01)

epidemic_simulator.set_statuses(statuses)
# for use later initializing the master equations

#for graphing against against users
Scount=len([node for node in user_nodes if statuses[node] == 'S'])
Ecount=len([node for node in user_nodes if statuses[node] == 'E'])
Icount=len([node for node in user_nodes if statuses[node] == 'I'])
Hcount=len([node for node in user_nodes if statuses[node] == 'H'])
Rcount=len([node for node in user_nodes if statuses[node] == 'R'])
Dcount=len([node for node in user_nodes if statuses[node] == 'D'])

time_trace = np.arange(time,simulation_length,static_contact_interval)
statuses_sum_trace = [[Scount,Ecount,Icount,Hcount,Rcount,Dcount]]


fig, axes = plt.subplots(1, 3, figsize = (16, 4))
# First we run and save the epidemic
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

    #statuses of user base (here Full)
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

plt.savefig('backward_forward_filter_on_loaded_epidemic.png', rasterized=True, dpi=150)


#
# Reset the world-time to 0, load the initial network
#

#
# Set the size of backward and forward DA windows
# For each cycle, this example performs a backward DA, a forward DA, and then a forward prediction 
# Current example only works with same size of intervals for backward/forward DA and prediction
# JW: I will further generalize the implementation
# 
backward_DA_interval = 1
forward_DA_interval = 1
forward_prediction_interval = 1

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

states_ensemble = deterministic_risk(population,
                                     initial_statuses,
                                     ensemble_size = ensemble_size)

master_eqn_ensemble.set_states_ensemble(states_ensemble)

forward_run_time = time
master_eqn_ensemble.set_start_time(time)
for j in range(int(backward_DA_interval/static_contact_interval)):

    loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=forward_run_time)
    user_network = loaded_data.contact_network.build_user_network_using(FullUserGraphBuilder())
    user_nodes = user_network.get_nodes()
    master_eqn_ensemble.set_mean_contact_duration(loaded_data.contact_network.get_edge_weights())
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)

    forward_run_time = forward_run_time + static_contact_interval
    master_eqn_ensemble.set_states_ensemble(states_ensemble)
    master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
                                       new_transmission_rate = community_transmission_rate_ensemble)
    states_trace_ensemble[:,:,j] = states_ensemble

#states_trace_ensemble[:,:,0] = states_ensemble

for k in range(1,int(simulation_length/backward_DA_interval)):
    backward_DA_time = k*backward_DA_interval
    master_eqn_ensemble.set_start_time(backward_DA_time)
    for i in range(int(backward_DA_interval/static_contact_interval)):
        loaded_data=epidemic_data_storage.get_network_from_end_time(end_time=backward_DA_time)
        user_network = loaded_data.contact_network.build_user_network_using(FullUserGraphBuilder())
        user_nodes = user_network.get_nodes()
        master_eqn_ensemble.set_mean_contact_duration(loaded_data.contact_network.get_edge_weights())
        states_ensemble = master_eqn_ensemble.simulate_backwards(static_contact_interval, n_steps = 25)
        
    
        (states_ensemble,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
        ) = assimilator_imperfect_observations.update(ensemble_state = states_ensemble,
                                                                data = loaded_data.start_statuses,
                                      full_ensemble_transition_rates = transition_rates_ensemble,
                                     full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                                          user_nodes = user_nodes)

        (states_ensemble,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
        ) = assimilator_perfect_observations.update(ensemble_state = states_ensemble,
                                                              data = loaded_data.start_statuses,
                                    full_ensemble_transition_rates = transition_rates_ensemble,
                                   full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                                        user_nodes = user_nodes)
        
        #update model parameters (transition and transmission rates) of the master eqn model
        
        #at the update the time
        backward_DA_time = backward_DA_time - static_contact_interval
        master_eqn_ensemble.set_states_ensemble(states_ensemble)
        master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
                                           new_transmission_rate = community_transmission_rate_ensemble)

    forward_DA_time = backward_DA_time
    master_eqn_ensemble.set_start_time(forward_DA_time)
    for j in range(int(forward_DA_interval/static_contact_interval)):
    
        loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=forward_DA_time)
        user_network = loaded_data.contact_network.build_user_network_using(FullUserGraphBuilder())
        user_nodes = user_network.get_nodes()
        master_eqn_ensemble.set_mean_contact_duration(loaded_data.contact_network.get_edge_weights())
        states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)
    
    
        (states_ensemble,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
        ) = assimilator_imperfect_observations.update(ensemble_state = states_ensemble,
                                                                data = loaded_data.end_statuses,
                                      full_ensemble_transition_rates = transition_rates_ensemble,
                                     full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                                        user_nodes = user_nodes)

        (states_ensemble,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
        ) = assimilator_perfect_observations.update(ensemble_state = states_ensemble,
                                                              data = loaded_data.end_statuses,
                                    full_ensemble_transition_rates = transition_rates_ensemble,
                                   full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                                        user_nodes = user_nodes)
    
        #update model parameters (transition and transmission rates) of the master eqn model
    
        #at the update the time
        forward_DA_time = forward_DA_time + static_contact_interval
        master_eqn_ensemble.set_states_ensemble(states_ensemble)
        master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
                                           new_transmission_rate = community_transmission_rate_ensemble)

    forward_prediction_time = forward_DA_time
    master_eqn_ensemble.set_start_time(forward_prediction_time)
    for j in range(int(forward_prediction_interval/static_contact_interval)):
    
        loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=forward_prediction_time)
        user_network = loaded_data.contact_network.build_user_network_using(FullUserGraphBuilder())
        user_nodes = user_network.get_nodes()
        master_eqn_ensemble.set_mean_contact_duration(loaded_data.contact_network.get_edge_weights())
        states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)
    
        #update model parameters (transition and transmission rates) of the master eqn model
    
        #at the update the time
        forward_prediction_time = forward_prediction_time + static_contact_interval
        master_eqn_ensemble.set_states_ensemble(states_ensemble)

        states_trace_ensemble[:,:,int(k*backward_DA_interval/static_contact_interval)+j] = states_ensemble

axes = plot_ensemble_states(states_trace_ensemble,
                            time_trace,
                            axes = axes,
                            xlims = (-0.1, simulation_length),
                            a_min = 0.0)
    
plt.savefig('backward_forward_filter_on_loaded_epidemic.png', rasterized=True, dpi=150)
