#load the initialization for the epidemic etc.
from _epidemic_initializer import *

from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
from epiforecast.user_base import FullUserGraphBuilder, ContiguousUserGraphBuilder
from epiforecast.measurements import Observation, DataObservation, HighVarianceObservation
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
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


#
# Run the epidemic simulation and store the results
#
start_time = epidemic_simulator.time
time = start_time
simulation_length = 20
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

#for graphing against against user_nodes
user_factor=population/user_population
Scount=user_factor*len([node for node in user_nodes if statuses[node] == 'S'])
Ecount=user_factor*len([node for node in user_nodes if statuses[node] == 'E'])
Icount=user_factor*len([node for node in user_nodes if statuses[node] == 'I'])
Hcount=user_factor*len([node for node in user_nodes if statuses[node] == 'H'])
Rcount=user_factor*len([node for node in user_nodes if statuses[node] == 'R'])
Dcount=user_factor*len([node for node in user_nodes if statuses[node] == 'D'])

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
    Scount=user_factor*len([node for node in user_nodes if statuses[node] == 'S'])
    Ecount=user_factor*len([node for node in user_nodes if statuses[node] == 'E'])
    Icount=user_factor*len([node for node in user_nodes if statuses[node] == 'I'])
    Hcount=user_factor*len([node for node in user_nodes if statuses[node] == 'H'])
    Rcount=user_factor*len([node for node in user_nodes if statuses[node] == 'R'])
    Dcount=user_factor*len([node for node in user_nodes if statuses[node] == 'D'])

    statuses_sum_trace.append([Scount,Ecount,Icount,Hcount,Rcount,Dcount])

axes = plot_epidemic_data(kinetic_model = epidemic_simulator.kinetic_model,
                          statuses_list = statuses_sum_trace,
                                   axes = axes,
                             plot_times = time_trace)

plt.savefig('filter_on_loaded_epidemic.png', rasterized=True, dpi=150)


#
# Reset the world-time to 0, load the initial network
#

time = 0.0
loaded_data = epidemic_data_storage.get_network_from_start_time(start_time=time)
user_network = loaded_data.contact_network.build_user_network_using(FullUserGraphBuilder())
#user_fraction = 0.1
#user_network= loaded_data.contact_network.build_user_network_using(FractionalUserGraphBuilder(user_fraction))
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

imperfect_observations=[high_var_infection_test]


negative_hospital_records = DataObservation(N = user_population,
                                    set_to_one=False,
                                    obs_status = 'H',
                                    obs_name = "nohospstate")

negative_death_records = DataObservation(N = user_population,
                                    set_to_one=False,
                                    obs_status = 'D',
                                    obs_name = "nodeathstate")

positive_hospital_records = DataObservation(N = user_population,
                                       set_to_one=True,
                                       obs_status = 'H',
                                       obs_name = "hospstate")

positive_death_records = DataObservation(N = user_population,
                                    set_to_one=True,
                                    obs_status = 'D',
                                    obs_name = "deathstate")

perfect_observations=[positive_death_records,
                      negative_death_records,
                      positive_hospital_records,
                      negative_hospital_records]

# create the assimilators
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
states_ensemble,_ = random_risk(population,
                                fraction_infected = 0.01,
                                ensemble_size = ensemble_size)

#loaded_data = epidemic_data_storage.get_network_from_start_time(start_time = time)
#statuses = loaded_data.start_statuses

# states_ensemble,_ = deterministic_risk(population,
#                                        statuses,
#                                        ensemble_size = ensemble_size)

master_eqn_ensemble.set_states_ensemble(states_ensemble)

states_trace_ensemble=np.zeros([ensemble_size,5*user_population,time_trace.size])
states_trace_ensemble[:,:,0] = states_ensemble

for i in range(int(simulation_length/static_contact_interval)):

    loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=time)
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
    time = time + static_contact_interval
    master_eqn_ensemble.set_states_ensemble(states_ensemble)
    master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
                                        new_transmission_rate = community_transmission_rate_ensemble)

    states_trace_ensemble[:,:,i] = states_ensemble

axes = plot_ensemble_states(states_trace_ensemble,
                            time_trace,
                            axes = axes,
                            xlims = (-0.1, simulation_length),
                            a_min = 0.0)
    
plt.savefig('filter_on_loaded_epidemic.png', rasterized=True, dpi=150)

