#load the initialization for the epidemic etc.
from _epidemic_initializer import *

from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.user_base import FullUserGraphBuilder
from epiforecast.measurements import Observation, DataObservation, HighVarianceObservation
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.scenarios import random_epidemic

from epiforecast.risk_simulator_initial_conditions import random_risk
from epiforecast.epiplots import plot_ensemble_states, plot_epidemic_data
from epiforecast.utilities import compartments_count

#
# create the  user_network (we do this here for plotting the epidemic)
#
user_network = network.build_user_network_using(FullUserGraphBuilder())

user_nodes = user_network.get_nodes()
user_population = user_network.get_node_count()

start_time = epidemic_simulator.time
simulation_length = 10 

print("We first create an epidemic for",
      simulation_length,
      "days")

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

statuses_all = []

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
    statuses_all.append(statuses)

    #statuses of user base
    user_statuses = { node : statuses[node] for node in user_nodes }
    n_S, n_E, n_I, n_H, n_R, n_D = compartments_count(user_statuses)
    statuses_sum_trace.append([n_S, n_E, n_I, n_H, n_R, n_D])

fig, axes = plt.subplots(1, 3, figsize = (16, 4))
axes = plot_epidemic_data(population = population,
                       statuses_list = statuses_sum_trace,
                                axes = axes,
                          plot_times = time_trace)

if not os.path.exists('data'):
    os.makedirs('data')

pickle.dump(statuses_all, open('data/epidemic_statuses_all.pkl', 'wb'))
plt.savefig('backward_forward_filter_on_loaded_epidemic.png', rasterized=True, dpi=150)

#
# Reset the world-time to 0, load the initial network
#

#
# Set the size of backward and forward DA windows
# For each cycle, this example performs a backward DA, a forward DA, and then a forward prediction 
# 
backward_DA_interval = 3
forward_DA_interval = 3
forward_prediction_interval = 1

start_time = 0.0 
loaded_data = epidemic_data_storage.get_network_from_start_time(start_time=start_time)
user_network = loaded_data.contact_network.build_user_network_using(FullUserGraphBuilder())
initial_statuses = loaded_data.start_statuses
n_user_nodes = user_network.get_node_count()

#
# Set up the population priors
#

ensemble_size = 100

transition_rates_ensemble = []

#all parameters should be positive, so we apply lognormal transform
for i in range(ensemble_size):
    transition_rates_member = TransitionRates.from_samplers(
               population = user_network.get_node_count(),
               lp_sampler = np.random.normal(0, 1, n_user_nodes),
              cip_sampler = np.random.normal(0, 1, n_user_nodes),
              hip_sampler = np.random.normal(0, 1, n_user_nodes),
               hf_sampler = hospitalization_fraction,
              cmf_sampler = community_mortality_fraction,
              hmf_sampler = hospital_mortality_fraction,
distributional_parameters = user_network.get_age_groups(),
             lp_transform = 'log',
            cip_transform = 'log',
            hip_transform = 'log')
    transition_rates_member.calculate_from_clinical()

    transition_rates_ensemble.append(transition_rates_member)


lp = np.array([rate.latent_periods for rate in transition_rates_ensemble])
cip = np.array([rate.community_infection_periods for rate in transition_rates_ensemble])
hip = np.array([rate.hospital_infection_periods for rate in transition_rates_ensemble])
print("latent_periods             : mean", np.mean(np.exp(lp)),  "var", np.var(np.exp(lp)))
print("community_infection_periods: mean", np.mean(np.exp(cip)), "var", np.var(np.exp(cip)))
print("hospital_infection_periods : mean", np.mean(np.exp(hip)), "var", np.var(np.exp(hip)))
#set transmission_rates
community_transmission_rate_ensemble = community_transmission_rate * np.ones([ensemble_size,1])

transition_rates_to_update_imperf_str = ['latent_periods',
                                  'community_infection_periods',
                                  'hospital_infection_periods']
rates_inflation = [0.0 ,0.0, 0.0] #sd of noise to inflate parameter with
transmission_rate_to_update_imperf_flag = False

transition_rates_to_update_perf_str = []
transmission_rate_to_update_perf_flag = False 


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
                                           obs_frac = 0.02,
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

imperfect_observations=[high_var_infection_test]

perfect_observations=[positive_hospital_records,
                      positive_death_records,
                      negative_hospital_records,
                      negative_death_records]

# create the assimilator
assimilator_imperfect_observations = DataAssimilator(observations = imperfect_observations,
                                                           errors = [],
                                   transition_rates_to_update_str = transition_rates_to_update_imperf_str,
                                 transmission_rate_to_update_flag = transmission_rate_to_update_imperf_flag)

assimilator_perfect_observations = DataAssimilator(observations = perfect_observations,
                                                         errors = [],
                                 transition_rates_to_update_str = transition_rates_to_update_perf_str,
                               transmission_rate_to_update_flag = transmission_rate_to_update_perf_flag)


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


states_ensemble = random_risk(population,
                              fraction_infected=0.01,
                              ensemble_size = ensemble_size)[0]


master_eqn_ensemble.set_states_ensemble(states_ensemble)

#
# Run backward/forward DA 
#
from _da_functions import *

(forward_run_end_time, 
states_ensemble,
states_ensemble_all,
master_eqn_ensemble, 
transition_rates_ensemble, community_transmission_rate_ensemble
) = forward_DA(start_time, forward_prediction_interval, 
                  day, static_contact_interval,
                  states_ensemble,
                  master_eqn_ensemble, 
                  epidemic_data_storage, user_network, user_nodes,
                  assimilator_imperfect_observations, assimilator_perfect_observations,
                  transition_rates_to_update_imperf_str, rates_inflation,
                  transition_rates_ensemble, community_transmission_rate_ensemble)

states_trace_ensemble[:,:,:int(forward_prediction_interval/static_contact_interval)] \
= states_ensemble_all

for k in range(1,int(simulation_length/forward_prediction_interval)):

    backward_start_time = k*forward_prediction_interval
    backward_interval_effective = np.minimum(backward_DA_interval, k*forward_prediction_interval)
    (backward_end_time, 
     states_ensemble,
     master_eqn_ensemble, 
     transition_rates_ensemble, community_transmission_rate_ensemble) \
    = backward_DA(backward_start_time, backward_interval_effective,
                                    day, static_contact_interval,
                                    master_eqn_ensemble, 
                                    epidemic_data_storage, user_network, user_nodes,
                                    assimilator_imperfect_observations, assimilator_perfect_observations,
                                    transition_rates_to_update_imperf_str, rates_inflation,
                                    transition_rates_ensemble, community_transmission_rate_ensemble)

    forward_DA_start_time = backward_end_time
    forward_DA_interval_effective = np.minimum(forward_DA_interval, k*forward_prediction_interval)
    (forward_DA_end_time,
     states_ensemble,
     states_ensemble_all,
     master_eqn_ensemble, 
     transition_rates_ensemble, community_transmission_rate_ensemble) \
    = forward_DA(forward_DA_start_time, forward_DA_interval_effective, 
                                    day, static_contact_interval,
                                    states_ensemble,
                                    master_eqn_ensemble, 
                                    epidemic_data_storage, user_network, user_nodes,
                                    assimilator_imperfect_observations, assimilator_perfect_observations,
                                    transition_rates_to_update_imperf_str, rates_inflation,
                                    transition_rates_ensemble, community_transmission_rate_ensemble)

    forward_run_start_time = forward_DA_end_time
    (forward_run_end_time,
     states_ensemble,
     states_ensemble_all,
     master_eqn_ensemble, 
     transition_rates_ensemble, community_transmission_rate_ensemble) \
    = forward_DA(forward_run_start_time, forward_prediction_interval, 
                                    day, static_contact_interval,
                                    states_ensemble,
                                    master_eqn_ensemble, 
                                    epidemic_data_storage, user_network, user_nodes,
                                    assimilator_imperfect_observations, assimilator_perfect_observations,
                                    transition_rates_to_update_imperf_str, rates_inflation,
                                    transition_rates_ensemble, community_transmission_rate_ensemble)

    states_trace_ensemble[:,:, \
                          int(k*forward_prediction_interval/static_contact_interval): \
                          int((k+1)*forward_prediction_interval/static_contact_interval)] \
                        = states_ensemble_all

pickle.dump(states_trace_ensemble, open('data/states_trace_ensemble.pkl', 'wb'))
pickle.dump(time_trace, open('data/time_trace.pkl', 'wb'))
axes = plot_ensemble_states(states_trace_ensemble,
                            time_trace,
                            axes = axes,
                            xlims = (-0.1, simulation_length),
                            a_min = 0.0)
    
plt.savefig('backward_forward_filter_on_loaded_epidemic.png', rasterized=True, dpi=150)
