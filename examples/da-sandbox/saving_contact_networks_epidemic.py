#load the initialization for the epidemic etc.
from _epidemic_initializer import *

from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.scenarios import random_epidemic
from epiforecast.risk_simulator_initial_conditions import deterministic_risk
from epiforecast.epiplots import plot_ensemble_states, plot_epidemic_data


start_time = epidemic_simulator.time
simulation_length = 2
print("We first create an epidemic for",
      simulation_length,
      "days, then we solve the master equations forward for this time")

# set up the initial conditions
fraction_infected = 0.01
statuses = random_epidemic(population,
                           populace,
                           fraction_infected)

epidemic_simulator.set_statuses(statuses)


time = start_time
time_trace = np.arange(time,simulation_length,static_contact_interval)
statuses_sum_trace = [[population-int(fraction_infected*population), 0, int(fraction_infected*population),0,0,0]]


# Run the epidemic simulation and store the results
for i in range(int(simulation_length/static_contact_interval)):
    network = epidemic_simulator.run(stop_time = epidemic_simulator.time + static_contact_interval,
                                     current_network = network)
    #save the start time network and statuses
    epidemic_data_storage.save_network_by_start_time(contact_network=network, start_time=time)
    epidemic_data_storage.save_start_statuses_to_network(start_time=time, start_statuses=statuses)

    #update the statuses and time
    statuses = epidemic_simulator.kinetic_model.current_statuses
    time = epidemic_simulator.time

    #save the statuses at the new time
    epidemic_data_storage.save_end_statuses_to_network(end_time=time, end_statuses=statuses)

    statuses_sum_trace.append([epidemic_simulator.kinetic_model.statuses['S'][-1],
                               epidemic_simulator.kinetic_model.statuses['E'][-1],
                               epidemic_simulator.kinetic_model.statuses['I'][-1],
                               epidemic_simulator.kinetic_model.statuses['H'][-1],
                               epidemic_simulator.kinetic_model.statuses['R'][-1],
                               epidemic_simulator.kinetic_model.statuses['D'][-1]]) 

fig, axes = plt.subplots(1, 3, figsize = (16, 4))
axes = plot_epidemic_data(kinetic_model = epidemic_simulator.kinetic_model,
                          statuses_list = statuses_sum_trace,
                                   axes = axes,
                             plot_times = time_trace)

plt.savefig('save_and_load_epidemic.png', rasterized=True, dpi=150)




#
# reset the time to the start of the simulation
#

time = start_time

#
# Set up the ensemble of master equtions
#

ensemble_size = 1 

transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(transition_rates)

#set transmission_rates
community_transmission_rate_ensemble = community_transmission_rate*np.ones([ensemble_size,1]) 

master_eqn_ensemble = MasterEquationModelEnsemble(population = population,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = community_transmission_rate_ensemble,
                                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                  ensemble_size = ensemble_size,
                                                  start_time = start_time)


#
# Run the master equations on the loaded networks
#
loaded_data = epidemic_data_storage.get_network_from_start_time(start_time = time)
statuses = loaded_data.start_statuses

states_ensemble,_ = deterministic_risk(population,
                                       statuses,
                                       ensemble_size = ensemble_size)

master_eqn_ensemble.set_states_ensemble(states_ensemble)

states_trace_ensemble=np.zeros([ensemble_size,5*population,time_trace.size])
states_trace_ensemble[:,:,0] = states_ensemble

for i in range(int(simulation_length/static_contact_interval)):

    loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=time)
    master_eqn_ensemble.set_mean_contact_duration(loaded_data.contact_network.get_edge_weights())
   
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

