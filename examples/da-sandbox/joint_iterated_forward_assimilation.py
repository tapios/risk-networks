import os, sys; sys.path.append(os.path.join('..', '..'))

from timeit import default_timer as timer
import numpy as np
from matplotlib import pyplot as plt
import copy 

from epiforecast.user_base import FullUserGraphBuilder
from epiforecast.forward_data_assimilator import DataAssimilator
from epiforecast.time_series import EnsembleTimeSeries
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
#from epiforecast.risk_simulator_initial_conditions import kinetic_to_master_same_fraction, random_risk_range
from epiforecast.epiplots import (plot_roc_curve, 
                                  plot_ensemble_states, 
                                  plot_epidemic_data,
                                  plot_transmission_rate, 
                                  plot_clinical_parameters)
from epiforecast.utilities import dict_slice, compartments_count
from epiforecast.populations import extract_ensemble_transition_rates

def get_start_time(start_end_time):
    return start_end_time.start

################################################################################
# initialization ###############################################################
################################################################################

# arguments parsing ############################################################
from _argparse_init import arguments

# constants ####################################################################
from _constants import (static_contact_interval,
                        start_time,
                        end_time,
                        total_time,
                        time_span,
                        distanced_max_contact_rate,
                        OUTPUT_PATH,
                        SEED_JOINT_EPIDEMIC)

# utilities ####################################################################
from _utilities import (print_info,
                        list_of_transition_rates_to_array,
                        modulo_is_close_to_zero,
                        are_close)

# general init #################################################################
import _general_init

# contact network ##############################################################
from _network_init import population, network

# stochastic model #############################################################
from _stochastic_init import (kinetic_ic, 
                              epidemic_simulator)

# user network #################################################################
from _user_network_init import user_network, user_nodes, user_population
# observations #################################################################
from _observations_init import (sensor_readings,
                                MDT_neighbor_test,
                                MDT_budget_random_test,
                                MDT_result_delay,
                                RDT_budget_random_test,
                                RDT_result_delay,
                                positive_hospital_records,
                                negative_hospital_records,
                                positive_death_records,
                                negative_death_records,
                                data_transform)


sensor_observations = [sensor_readings]

viral_test_observations = [RDT_budget_random_test]
test_result_delay = RDT_result_delay # delay to results of the virus test

#record_observations   = [positive_hospital_records,
#                          positive_death_records]

record_observations   = [positive_hospital_records,
                         negative_hospital_records,
                         positive_death_records,
                         negative_death_records]

# master equations #############################################################
from _master_eqn_init import (master_eqn_ensemble,
                              ensemble_size,
                              ensemble_ic,
                              transition_rates_ensemble,
                              community_transmission_rate_ensemble,
                              learn_transition_rates,
                              learn_transmission_rate,
                              parameter_str,
                              transition_rates_min,
                              transition_rates_max,
                              transmission_rate_min,
                              transmission_rate_max,
                              n_forward_steps,
                              n_backward_steps)

# assimilator ##################################################################
transition_rates_to_update_str   = parameter_str
transmission_rate_to_update_flag = learn_transmission_rate 



sensor_assimilator = DataAssimilator(
        observations=sensor_observations,
        errors=[],
        data_transform=data_transform,
        n_assimilation_batches = arguments.assimilation_batches_sensor,
        transition_rates_to_update_str=[],
        transmission_rate_to_update_flag=transmission_rate_to_update_flag,
        update_type=arguments.assimilation_update_sensor,
        joint_cov_noise=arguments.sensor_assimilation_joint_regularization,
        obs_cov_noise=arguments.sensor_assimilation_obs_regularization,
        full_svd=True,
        inflate_states=arguments.assimilation_inflation,
        inflate_reg=arguments.assimilation_sensor_inflation,
        inflate_I_only=arguments.assimilation_inflate_I_only,
        distance_threshold=arguments.distance_threshold,        
        transmission_rate_min=transmission_rate_min,
        transmission_rate_max=transmission_rate_max,
        transmission_rate_inflation=arguments.params_transmission_inflation,
        output_path=OUTPUT_PATH)

viral_test_assimilator = DataAssimilator(
        observations=viral_test_observations,
        errors=[],
        data_transform=data_transform,
        n_assimilation_batches = arguments.assimilation_batches_test,
        transition_rates_to_update_str=transition_rates_to_update_str,
        transmission_rate_to_update_flag=transmission_rate_to_update_flag,
        update_type=arguments.assimilation_update_test,
        joint_cov_noise=arguments.test_assimilation_joint_regularization,
        obs_cov_noise=arguments.test_assimilation_obs_regularization,
        full_svd=True,
        inflate_states=arguments.assimilation_inflation,
        inflate_reg=arguments.assimilation_test_inflation,
        inflate_I_only=arguments.assimilation_inflate_I_only,
        distance_threshold=arguments.distance_threshold,
        transition_rates_min=transition_rates_min,
        transition_rates_max=transition_rates_max,
        transmission_rate_min=transmission_rate_min,
        transmission_rate_max=transmission_rate_max,
        transmission_rate_inflation=arguments.params_transmission_inflation,
        output_path=OUTPUT_PATH)

record_assimilator = DataAssimilator(
        observations=record_observations,
        errors=[],
        data_transform=data_transform,
        HDflag=1,
        n_assimilation_batches=arguments.assimilation_batches_record,
        transition_rates_to_update_str=[],
        transmission_rate_to_update_flag=True,
        update_type=arguments.assimilation_update_record,
        joint_cov_noise=arguments.record_assimilation_joint_regularization,
        obs_cov_noise=arguments.record_assimilation_obs_regularization,
        full_svd=True,    
        inflate_states=arguments.assimilation_inflation,
        inflate_reg=arguments.assimilation_record_inflation,
        inflate_I_only=arguments.assimilation_inflate_I_only,
        distance_threshold=arguments.distance_threshold,
        transmission_rate_min=transmission_rate_min,
        transmission_rate_max=transmission_rate_max,
        transmission_rate_inflation=arguments.params_transmission_inflation,
        output_path=OUTPUT_PATH)

# post-processing ##############################################################
#from _post_process_init import axes
fig, axes = plt.subplots(1, 3, figsize = (16, 4))

# inverventions ################################################################
from _intervention_init import (intervention,
                                intervention_frequency,
                                intervention_nodes, 
                                intervention_type,
                                query_intervention,
                                intervention_sick_isolate_time) 

################################################################################
# epidemic setup ###############################################################
################################################################################

from _utilities import print_start_of, print_end_of, print_info_module
################################################################################
kinetic_state = kinetic_ic # dict { node : compartment }

user_state = dict_slice(kinetic_state, user_nodes)
n_S, n_E, n_I, n_H, n_R, n_D = compartments_count(user_state)
statuses_sum_trace = []
statuses_sum_trace.append([n_S, n_E, n_I, n_H, n_R, n_D])

kinetic_states_timeseries = []
kinetic_states_timeseries.append(kinetic_state) # storing ic

################################################################################
# master equations + data assimilation init ####################################
################################################################################
# constants ####################################################################

#floats
da_window         = arguments.assimilation_window
prediction_window = 1.0
save_to_file_interval = 1.0
sensor_assimilation_interval  = 1.0 # same for I
test_assimilation_interval  = 1.0 # same for I
record_assimilation_interval = 1.0 # assimilate H and D data every .. days

intervention_start_time = arguments.intervention_start_time
intervention_interval = arguments.intervention_interval
#ints
n_sweeps                     = arguments.assimilation_sweeps
n_record_sweeps              = 1
n_prediction_windows_spin_up = 8
n_prediction_windows         = int(total_time/prediction_window)
steps_per_da_window          = int(da_window/static_contact_interval)
steps_per_prediction_window  = int(prediction_window/static_contact_interval)

assert n_prediction_windows_spin_up * prediction_window + prediction_window > da_window
earliest_assimilation_time = (n_prediction_windows_spin_up + 1)* prediction_window - da_window 
assert n_prediction_windows > n_prediction_windows_spin_up


# epidemic storage #############################################################
# Set an upper limit on number of stored contact networks:
max_networks = steps_per_da_window + steps_per_prediction_window 
epidemic_data_storage = StaticIntervalDataSeries(static_contact_interval, max_networks=max_networks)

# storing ######################################################################
#for the initial run we smooth over a window, store data by time-stamp.
ensemble_state_series_dict = {} 

master_states_sum_timeseries  = EnsembleTimeSeries(ensemble_size,
                                                   6,
                                                   time_span.size)

mean_transmission_rate_timeseries = EnsembleTimeSeries(ensemble_size,
                                                       1,
                                                       time_span.size)

transition_rates_timeseries = EnsembleTimeSeries(ensemble_size,
                                              6,
                                              time_span.size)

# intial conditions  ###########################################################

master_eqn_ensemble.set_states_ensemble(ensemble_ic)
master_eqn_ensemble.set_start_time(start_time)

################################################################################
# master equations + data assimilation computation #############################
################################################################################
# spin-up w/o data assimilation ################################################
current_time = start_time
spin_up_steps = n_prediction_windows_spin_up * steps_per_prediction_window
prediction_steps = n_prediction_windows * steps_per_prediction_window
ensemble_state = ensemble_ic

timer_spin_up = timer()

print_info("Spin-up started")
for j in range(spin_up_steps):
    walltime_master_eqn = 0.0
    master_eqn_ensemble.reset_walltimes()
    #Run kinetic model
    # run
    KE_timer = timer()
    network = epidemic_simulator.run(
            stop_time=current_time + static_contact_interval,
            current_network=network)
    print("KE runtime", timer()-KE_timer, flush=True)
    # store for further usage (master equations etc)
    DS_timer = timer()
    epidemic_data_storage.save_network_by_start_time(
            start_time=current_time,
            contact_network=network)
    epidemic_data_storage.save_start_statuses_to_network(
            start_time=current_time,
            start_statuses=kinetic_state)
    
    # save kinetic data if required, note current time has advanced since saving ensemble state:
    save_kinetic_state_now = modulo_is_close_to_zero(current_time, 
                                                     save_to_file_interval, 
                                                     eps=static_contact_interval)
    if save_kinetic_state_now:
        kinetic_state_path = os.path.join(OUTPUT_PATH, 'kinetic_eqns_statuses_at_step_'+str(j)+'.npy')
        kinetic_eqns_statuses = dict_slice(kinetic_state, user_nodes)
        np.save(kinetic_state_path, kinetic_eqns_statuses)
  
    kinetic_state = epidemic_simulator.kinetic_model.current_statuses
    epidemic_data_storage.save_end_statuses_to_network(
            end_time=current_time+static_contact_interval,
            end_statuses=kinetic_state)
    print("network and data storage runtime",timer()-DS_timer,flush=True) 
    # store for plotting
    PS_timer = timer() 
    user_state = dict_slice(kinetic_state, user_nodes)
    n_S, n_E, n_I, n_H, n_R, n_D = compartments_count(user_state)
    statuses_sum_trace.append([n_S, n_E, n_I, n_H, n_R, n_D])

    kinetic_states_timeseries.append(kinetic_state)
    print("store KE statuses and timeseries runtime", timer() - PS_timer,flush=True)
  
  

    # now for the master eqn
    ensemble_state_frac = ensemble_state.reshape(ensemble_size, 6, -1).sum(axis = 2)/user_population
    master_states_sum_timeseries.push_back(ensemble_state_frac) # storage
    
    if learn_transmission_rate == True:
        mean_community_transmission_rate_ensemble = community_transmission_rate_ensemble.mean(axis=1)[:,np.newaxis]
        mean_transmission_rate_timeseries.push_back(
                mean_community_transmission_rate_ensemble)
    if learn_transition_rates == True:
        transition_rates_timeseries.push_back(
                extract_ensemble_transition_rates(transition_rates_ensemble))

    #save the ensemble if required - we do here are we do not save master eqn at end of DA-windows
    save_ensemble_state_now = modulo_is_close_to_zero(current_time - static_contact_interval, 
                                                      save_to_file_interval, 
                                                      eps=static_contact_interval)
    if save_ensemble_state_now:
        ensemble_state_path = os.path.join(OUTPUT_PATH, 'master_eqns_mean_states_at_step_'+str(j-1)+'.npy')
        master_eqns_mean_states = ensemble_state.mean(axis=0)
        np.save(ensemble_state_path,master_eqns_mean_states)
            
    loaded_data = epidemic_data_storage.get_network_from_start_time(
            start_time=current_time)

    user_network.update_from(loaded_data.contact_network)
    master_eqn_ensemble.set_mean_contact_duration(
            user_network.get_edge_weights())
    timer_master_eqn = timer()
    

    ensemble_state = master_eqn_ensemble.simulate(
        static_contact_interval,
        min_steps=n_forward_steps)
    #move to new time
    current_time += static_contact_interval
    current_time_span = [time for time in time_span if time < current_time+static_contact_interval]
    walltime_master_eqn += timer() - timer_master_eqn
    print_info("eval_closure walltime:", master_eqn_ensemble.get_walltime_eval_closure())
    print_info("master equations walltime:", walltime_master_eqn, end='\n\n')

    #in theory should do nothing
    master_eqn_ensemble.set_states_ensemble(ensemble_state)
    master_eqn_ensemble.set_start_time(current_time)

    #generate data to be assimilated later on
    if current_time > (earliest_assimilation_time - 0.1*static_contact_interval):
        observe_sensor_now = modulo_is_close_to_zero(current_time,
                                                   sensor_assimilation_interval,
                                                   eps=static_contact_interval)
        if observe_sensor_now:
            sensor_assimilator.find_and_store_observations(
                ensemble_state,
                loaded_data.end_statuses,
                user_network,
                current_time)
            

        observe_test_now = modulo_is_close_to_zero(current_time,
                                                   test_assimilation_interval,
                                                   eps=static_contact_interval)
        if observe_test_now:
            viral_test_assimilator.find_and_store_observations(
                ensemble_state,
                loaded_data.end_statuses,
                user_network,
                current_time)

        observe_record_now = modulo_is_close_to_zero(current_time,
                                                    record_assimilation_interval,
                                                    eps=static_contact_interval)
        if observe_record_now:
            record_assimilator.find_and_store_observations(
                ensemble_state,
                loaded_data.end_statuses,
                user_network,
                current_time)

        if observe_sensor_now or observe_test_now or observe_record_now:
            ensemble_state_series_dict[current_time] = copy.deepcopy(ensemble_state )


    #plots on the fly
    plot_and_save_now = modulo_is_close_to_zero(current_time - static_contact_interval, 
                                                save_to_file_interval, 
                                                eps=static_contact_interval)
    if plot_and_save_now:
        if (current_time - static_contact_interval) > static_contact_interval: # i.e not first step
            plt.close(fig)

            if learn_transmission_rate == True:
                plot_transmission_rate(mean_transmission_rate_timeseries.container[:,:, :len(current_time_span)-1],
                                       current_time_span[:-1],
                                       a_min=0.0,
                                       output_path=OUTPUT_PATH)
                

            fig, axes = plt.subplots(1, 3, figsize = (16, 4))
            axes = plot_epidemic_data(user_population, 
                                      statuses_sum_trace, 
                                      axes, 
                                      current_time_span)
          
            plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic.png'), rasterized=True, dpi=150)
            
            axes = plot_ensemble_states(user_population,
                                        population,
                                        master_states_sum_timeseries.container[:,:, :len(current_time_span)-1],
                                        current_time_span[:-1],
                                        axes=axes,
                                        xlims=(-0.1, current_time),
                                        a_min=0.0)
            plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic_and_master_eqn.png'),
                        rasterized=True,
                        dpi=150)


    #intervention if required
    intervene_now = query_intervention(intervention_frequency,current_time,intervention_start_time, static_contact_interval)    
    
    if intervene_now:
        
        # now see which nodes have intervention applied
        if intervention_nodes == "all":
            nodes_to_intervene = network.get_nodes()
            print("intervention applied to all {:d} nodes.".format(
                network.get_node_count()))
            
        elif intervention_nodes == "sick":
            nodes_to_intervene_current = intervention.find_sick(ensemble_state)
            intervention.save_nodes_to_intervene(current_time, 
                                                 nodes_to_intervene_current)
            nodes_to_intervene = \
                    np.unique( \
                    np.concatenate([v \
                    for k, v in intervention.stored_nodes_to_intervene.items() \
                    if k > current_time - intervention_sick_isolate_time]) \
                    )
            print("intervention applied to sick nodes: {:d}/{:d}".format(
                nodes_to_intervene.size, network.get_node_count()))
            #raise ValueError("Currently interventions only work for 'all', see below")
        else:
            raise ValueError("unknown 'intervention_nodes', choose from 'all' (default), 'sick'")

        # Apply the the chosen form of intervention
        if intervention_type == "isolate":
            network.isolate(nodes_to_intervene) 

        elif intervention_type == "social_distance":
            λ_min, λ_max = network.get_lambdas() #returns np.array (num_nodes,) for each lambda [Not a dict!]
            λ_max[:] = distanced_max_contact_rate 
            network.set_lambdas(λ_min,λ_max)

            λ_min, λ_max = user_network.get_lambdas() #returns np.aray( num_nodes,) [ not a dict!]
            λ_max[:] = distanced_max_contact_rate 
            user_network.set_lambdas(λ_min,λ_max)

        else:
            raise ValueError("unknown intervention type, choose from 'social_distance' (default), 'isolate' ")

    
        
print_info("Spin-up ended; elapsed:", timer() - timer_spin_up, end='\n\n')
print_info("Spin-up ended: current time", current_time)

# main loop: backward/forward/data assimilation ################################
# 3 stages per loop:
# 1a) run epidemic for the duration of the prediction window 
# 1b) prediction (no assimilation) forwards steps_per_prediction_window
#    - Save data during this window from [ start , end ]
#    - Make observations and store them in the assimilator
# 3) Assimilation update at start of window, using recorded data over the window [start,end]
#    - rerun master equations over window
#    - possibly repeat 3) over n_sweeps
# Repeat from 1)
#

for k in range(n_prediction_windows_spin_up, n_prediction_windows):
    print_info("Prediction window: {}/{}".format(k+1, n_prediction_windows))
    timer_window = timer()
    walltime_master_eqn = 0.0
    walltime_DA_update = 0.0
    master_eqn_ensemble.reset_walltimes()

    assert are_close(current_time,
                     k * prediction_window,
                     eps=static_contact_interval)
    current_time = k * prediction_window # to avoid build-up of errors

    ensemble_state_frac = ensemble_state.reshape(ensemble_size, 6, -1).sum(axis = 2)/user_population
    print(current_time, ensemble_state_frac.mean(axis=0))
    
    ## 1a) Run epidemic simulator
    ## 1b) forward run w/o data assimilation; prediction
    print("Start time = ", current_time)
    master_eqn_ensemble.set_start_time(current_time)
    for j in range(steps_per_prediction_window):
                
        # run epidemic_simulator
        network = epidemic_simulator.run(
            stop_time=current_time + static_contact_interval,
            current_network=network)

        # store for further usage (master equations etc)
        epidemic_data_storage.save_network_by_start_time(
            start_time=current_time,
            contact_network=network)
        epidemic_data_storage.save_start_statuses_to_network(
            start_time=current_time,
            start_statuses=kinetic_state)

        # save kinetic data if required:
        save_kinetic_state_now = modulo_is_close_to_zero(current_time, 
                                                         save_to_file_interval,                                                     
                                                         eps=static_contact_interval)
        if save_kinetic_state_now:
            kinetic_state_path = os.path.join(OUTPUT_PATH, 'kinetic_eqns_statuses_at_step_'+str(k*steps_per_prediction_window+j)+'.npy')
            kinetic_eqns_statuses = dict_slice(kinetic_state, user_nodes)
            np.save(kinetic_state_path, kinetic_eqns_statuses)
        
        kinetic_state = epidemic_simulator.kinetic_model.current_statuses
        epidemic_data_storage.save_end_statuses_to_network(
            end_time=current_time+static_contact_interval,
            end_statuses=kinetic_state)

        # store for plotting
        user_state = dict_slice(kinetic_state, user_nodes)
        n_S, n_E, n_I, n_H, n_R, n_D = compartments_count(user_state)
        statuses_sum_trace.append([n_S, n_E, n_I, n_H, n_R, n_D])
        
        kinetic_states_timeseries.append(kinetic_state)

        # storage of data first (we do not store end of prediction window)
        ensemble_state_frac = ensemble_state.reshape(ensemble_size, 6, -1).sum(axis = 2)/user_population
        master_states_sum_timeseries.push_back(ensemble_state_frac) # storage

        if learn_transmission_rate == True:
            mean_community_transmission_rate_ensemble = community_transmission_rate_ensemble.mean(axis=1)[:,np.newaxis]
            mean_transmission_rate_timeseries.push_back(
                    mean_community_transmission_rate_ensemble)
        if learn_transition_rates == True:
            transition_rates_timeseries.push_back(
                    extract_ensemble_transition_rates(transition_rates_ensemble))

        #save the ensemble if required - we do here are we do not save master eqn at end of DA-windows
        save_ensemble_state_now = modulo_is_close_to_zero(current_time - static_contact_interval, 
                                                          save_to_file_interval, 
                                                          eps=static_contact_interval)
        if save_ensemble_state_now:
            ensemble_state_path = os.path.join(OUTPUT_PATH, 'master_eqns_mean_states_at_step_'+str(k*steps_per_prediction_window+j-1)+'.npy')
            master_eqns_mean_states = ensemble_state.mean(axis=0)
            np.save(ensemble_state_path,master_eqns_mean_states)
                
        loaded_data = epidemic_data_storage.get_network_from_start_time(start_time=current_time)

        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(user_network.get_edge_weights())

        # run ensemble forward
        timer_master_eqn = timer()
        
        ensemble_state = master_eqn_ensemble.simulate(static_contact_interval,
                                                      min_steps=n_forward_steps)

        walltime_master_eqn += timer() - timer_master_eqn

        current_time += static_contact_interval
        current_time_span = [time for time in time_span if time < current_time+static_contact_interval]
   
        # collect data for later assimilation
        observe_sensor_now = modulo_is_close_to_zero(current_time,
                                                     sensor_assimilation_interval,
                                                     eps=static_contact_interval)

        if observe_sensor_now:
            sensor_assimilator.find_and_store_observations(
                ensemble_state,
                loaded_data.end_statuses,
                user_network,
                current_time)

        observe_test_now = modulo_is_close_to_zero(current_time,
                                                   test_assimilation_interval,
                                                   eps=static_contact_interval)
        if observe_test_now:
            viral_test_assimilator.find_and_store_observations(
                ensemble_state,
                loaded_data.end_statuses,
                user_network,
                current_time)

        observe_record_now = modulo_is_close_to_zero(current_time,
                                                    record_assimilation_interval,
                                                    eps=static_contact_interval)
        if observe_record_now:
            record_assimilator.find_and_store_observations(
                ensemble_state,
                loaded_data.end_statuses,
                user_network,
                current_time)
        
        if observe_sensor_now or observe_test_now or observe_record_now:
            ensemble_state_series_dict[current_time] = copy.deepcopy(ensemble_state)


        #plots on the fly    
        plot_and_save_now = modulo_is_close_to_zero(current_time - static_contact_interval, 
                                                    save_to_file_interval, 
                                                    eps=static_contact_interval)
        if plot_and_save_now:
            plt.close(fig)
            
            if learn_transmission_rate == True:
                plot_transmission_rate(mean_transmission_rate_timeseries.container[:,:, :len(current_time_span)-1],
                                       current_time_span[:-1],
                                       a_min=0.0,
                                       output_path=OUTPUT_PATH)
                

            fig, axes = plt.subplots(1, 3, figsize = (16, 4))
            axes = plot_epidemic_data(user_population, 
                                      statuses_sum_trace, 
                                      axes, 
                                      current_time_span)
            plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic.png'), rasterized=True, dpi=150)
            

            # plot trajectories
            axes = plot_ensemble_states(user_population,
                                        population,
                                        master_states_sum_timeseries.container[:,:, :len(current_time_span)-1],
                                        current_time_span[:-1],
                                        axes=axes,
                                        xlims=(-0.1, current_time),
                                        a_min=0.0)
            plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic_and_master_eqn.png'),
                        rasterized=True,
                        dpi=150)
            
            
    print_info("Prediction ended: current time:", current_time)
    for step in range((2+n_record_sweeps)*n_sweeps):
         # by restarting from time of first assimilation data
        past_time = current_time - steps_per_da_window * static_contact_interval
    
        if step == 0:
            # remove the earliest dictionaries
            stored_states_times = ensemble_state_series_dict.keys() 
            times_for_removal = [time for time in stored_states_times if time < past_time]
            [ensemble_state_series_dict.pop(time) for time in times_for_removal]
        
        DA_update_timer = timer()
        # DA update of initial state IC and parameters at t0, due to data collected in window [t0,t1]
        if step % (2+n_record_sweeps) == 0:
            (ensemble_state_series_dict, 
             transition_rates_ensemble,
             community_transmission_rate_ensemble,
             update_flag
            ) = sensor_assimilator.update_initial_from_series(
             ensemble_state_series_dict, 
             transition_rates_ensemble,
             community_transmission_rate_ensemble,
             user_network)       
            print("assimilated sensors",flush=True)
        elif step % (2+n_record_sweeps) == 1:
            (ensemble_state_series_dict, 
             transition_rates_ensemble,
             community_transmission_rate_ensemble,
             update_flag
            ) = viral_test_assimilator.update_initial_from_series(
             ensemble_state_series_dict, 
             transition_rates_ensemble,
             community_transmission_rate_ensemble,
             user_network)
             
            print("assimilated viral tests",flush=True)
        elif step % (2+n_record_sweeps) >= 2:
            (ensemble_state_series_dict, 
             transition_rates_ensemble,
             community_transmission_rate_ensemble,
             update_flag
         ) = record_assimilator.update_initial_from_series(
             ensemble_state_series_dict, 
             transition_rates_ensemble,
             community_transmission_rate_ensemble,
             user_network)

            print("assimilated records",flush=True)
            
        # run ensemble of master equations again over the da windowprediction loop again without data collection
        walltime_DA_update += timer() - DA_update_timer
        
        if update_flag:
            # update with the new initial state and parameters 
            master_eqn_ensemble.set_states_ensemble(ensemble_state_series_dict[past_time])
            master_eqn_ensemble.set_start_time(past_time)
            master_eqn_ensemble.update_ensemble(
                new_transition_rates=transition_rates_ensemble,
                new_transmission_rate_parameters=community_transmission_rate_ensemble)
            
            ensemble_state_frac = ensemble_state_series_dict[past_time].reshape(ensemble_size, 6, -1).sum(axis = 2)/user_population
#            print(past_time, np.var(ensemble_state[:,982:2*982], axis=0))
                    
            for j in range(steps_per_da_window):

                walltime_master_eqn = 0.0
                master_eqn_ensemble.reset_walltimes()
                # load the new network
                loaded_data = epidemic_data_storage.get_network_from_start_time(
                    start_time=past_time)
                
                user_network.update_from(loaded_data.contact_network)
                master_eqn_ensemble.set_mean_contact_duration(
                    user_network.get_edge_weights())
                timer_master_eqn = timer()
                
                # simulate the master equations
                ensemble_state = master_eqn_ensemble.simulate(
                    static_contact_interval,
                    min_steps=n_forward_steps)
                
                # move to new time
                past_time += static_contact_interval
                walltime_master_eqn += timer() - timer_master_eqn
         
                # overwrite the data.
                observe_sensor_now = modulo_is_close_to_zero(past_time,
                                                             sensor_assimilation_interval,
                                                             eps=static_contact_interval)
                
                observe_test_now = modulo_is_close_to_zero(past_time,
                                                           test_assimilation_interval,
                                                           eps=static_contact_interval)
                
                observe_record_now = modulo_is_close_to_zero(past_time,
                                                             record_assimilation_interval,
                                                             eps=static_contact_interval)
                
                if observe_sensor_now or observe_test_now or observe_record_now:
                    ensemble_state_series_dict[past_time] =copy.deepcopy( ensemble_state )
                
            print("Completed forward sweep iteration {}/{}".format(step + 1, 3*n_sweeps), 
                  " over the interval [{},{}]".format(current_time - steps_per_da_window * static_contact_interval, past_time)) 
            # DA should get back to the current time
            assert are_close(past_time, current_time, eps=static_contact_interval)


        else:
            print("Completed forward sweep iteration {}/{}".format(step+1,3*n_sweeps), ", no forward sweep required")
        

    print_info("Prediction window: {}/{}".format(k+1, n_prediction_windows),
               "ended; elapsed:",
               timer() - timer_window)
    
    print_info("Prediction window: {}/{};".format(k+1, n_prediction_windows),
               "eval_closure walltime:",
               master_eqn_ensemble.get_walltime_eval_closure())
            
    print_info("Prediction window: {}/{};".format(k+1, n_prediction_windows),
               "master equations walltime:",
               walltime_master_eqn, end='\n\n')
    
    print_info("Prediction window: {}/{};".format(k+1, n_prediction_windows),
               "assimilator(s) walltime:",
               walltime_DA_update, end='\n\n')
            
    #4) Intervention
    intervene_now = query_intervention(intervention_frequency,current_time,intervention_start_time, static_contact_interval)    
            
    if intervene_now:
        # now see which nodes have intervention applied
        if intervention_nodes == "all":
            nodes_to_intervene = network.get_nodes() 
            print("intervention applied to all {:d} nodes".format(
                network.get_node_count()))
            
        elif intervention_nodes == "sick":
            nodes_to_intervene_current = intervention.find_sick(ensemble_state)
            intervention.save_nodes_to_intervene(current_time, 
                                                 nodes_to_intervene_current)
            nodes_to_intervene = \
                    np.unique( \
                    np.concatenate([v \
                    for k, v in intervention.stored_nodes_to_intervene.items() \
                    if k > current_time - intervention_sick_isolate_time]) \
                    )
            print("intervention applied to sick nodes: {:d}/{:d}".format(
                nodes_to_intervene.size, network.get_node_count()))
            #raise ValueError("Currently interventions only work for 'all'")
        else:
            raise ValueError("unknown 'intervention_nodes', choose from 'all' (default), 'sick'")
            
        # Apply the the chosen form of intervention
        if intervention_type == "isolate":
            network.isolate(nodes_to_intervene) 
            
        elif intervention_type == "social_distance":
            λ_min, λ_max = network.get_lambdas() #returns np.array (num_nodes,) for each lambda [Not a dict!]
            λ_max[:] = distanced_max_contact_rate 
            network.set_lambdas(λ_min,λ_max)

            λ_min, λ_max = user_network.get_lambdas() #returns np.aray( num_nodes,) [ not a dict!]
            λ_max[:] = distanced_max_contact_rate 
            user_network.set_lambdas(λ_min,λ_max)

        else:
            raise ValueError("unknown intervention type, choose from 'social_distance' (default), 'isolate' ")



## Final storage after last step
ensemble_state_frac = ensemble_state.reshape(ensemble_size, 6, -1).sum(axis = 2)/user_population
master_states_sum_timeseries.push_back(ensemble_state_frac) # storage

if learn_transmission_rate == True:
    mean_community_transmission_rate_ensemble = community_transmission_rate_ensemble.mean(axis=1)[:,np.newaxis]
    mean_transmission_rate_timeseries.push_back(
            mean_community_transmission_rate_ensemble)
if learn_transition_rates == True:
    transition_rates_timeseries.push_back(
            extract_ensemble_transition_rates(transition_rates_ensemble))


## Final save after last step
save_kinetic_state_now = modulo_is_close_to_zero(current_time, 
                                                 save_to_file_interval,                                                     
                                                 eps=static_contact_interval)
if save_kinetic_state_now:
    kinetic_state_path = os.path.join(OUTPUT_PATH, 'kinetic_eqns_statuses_at_step_'+str(prediction_steps)+'.npy')
    kinetic_eqns_statuses = dict_slice(kinetic_state, user_nodes)
    np.save(kinetic_state_path, kinetic_eqns_statuses)

save_ensemble_state_now = modulo_is_close_to_zero(current_time,
                                                  save_to_file_interval,  
                                                  eps=static_contact_interval)
if save_ensemble_state_now:
    ensemble_state_path = os.path.join(OUTPUT_PATH, 'master_eqns_mean_states_at_step_'+str(prediction_steps)+'.npy')
    master_eqns_mean_states = ensemble_state.mean(axis=0)
    np.save(ensemble_state_path,master_eqns_mean_states)


print("finished assimilation")
# save & plot ##################################################################
plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize = (16, 4))
axes = plot_epidemic_data(user_population, statuses_sum_trace, axes, time_span)
plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic.png'), rasterized=True, dpi=150)

# plot trajectories
axes = plot_ensemble_states(user_population,
                            population,
                            master_states_sum_timeseries.container,
                            time_span,
                            axes=axes,
                            xlims=(-0.1, total_time),
                            a_min=0.0)
plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic_and_master_eqn.png'),
            rasterized=True,
            dpi=150)

plt.close()

np.save(os.path.join(OUTPUT_PATH, 'trace_kinetic_statuses_sum.npy'), 
        statuses_sum_trace)


np.save(os.path.join(OUTPUT_PATH, 'trace_master_states_sum.npy'), 
        master_states_sum_timeseries.container)

np.save(os.path.join(OUTPUT_PATH, 'time_span.npy'), 
        time_span)
np.save(os.path.join(OUTPUT_PATH, 'user_nodes.npy'), 
        user_nodes)


if learn_transmission_rate == True:
    plot_transmission_rate(mean_transmission_rate_timeseries.container,
            time_span,
            a_min=0.0,
            output_path=OUTPUT_PATH)

if learn_transition_rates == True:
    plot_clinical_parameters(transition_rates_timeseries.container,
            time_span,
            a_min=0.0,
            output_path=OUTPUT_PATH)

# save parameters ################################################################
if learn_transmission_rate == True:
    np.save(os.path.join(OUTPUT_PATH, 'transmission_rate.npy'), 
            mean_transmission_rate_timeseries.container)

if learn_transition_rates == True:
    np.save(os.path.join(OUTPUT_PATH, 'transition_rates.npy'), 
            transition_rates_timeseries.container)

np.save(os.path.join(OUTPUT_PATH, 'master_eqns_states_sum.npy'), master_states_sum_timeseries.container) #save the ensemble fracs for graphing

#kinetic_eqns_statuses = []
#for kinetic_state in kinetic_states_timeseries:
#    kinetic_eqns_statuses.append(dict_slice(kinetic_state, user_nodes))

#np.save(os.path.join(OUTPUT_PATH, 'kinetic_eqns_statuses.npy'), kinetic_eqns_statuses)


