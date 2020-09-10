import os, sys; sys.path.append(os.path.join('..', '..'))

from timeit import default_timer as timer
import numpy as np
from matplotlib import pyplot as plt

from epiforecast.user_base import FullUserGraphBuilder
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.time_series import EnsembleTimeSeries
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
from epiforecast.risk_simulator_initial_conditions import kinetic_to_master_same_fraction, random_risk_range
from epiforecast.epiplots import plot_roc_curve, plot_ensemble_states, plot_epidemic_data
from epiforecast.utilities import dict_slice, compartments_count

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
                                negative_death_records)


sensor_observations = [sensor_readings]

viral_test_observations = [MDT_neighbor_test]
test_result_delay = MDT_result_delay # delay to results of the virus test

record_observations   = [positive_hospital_records,
                          negative_hospital_records,
                          positive_death_records,
                          negative_death_records]


# assimilator ##################################################################
transition_rates_to_update_str   = []
transmission_rate_to_update_flag = False

sensor_assimilator = DataAssimilator(
        observations=sensor_observations,
        errors=[],
        n_assimilation_batches = arguments.assimilation_batches_sensor,
        transition_rates_to_update_str=transition_rates_to_update_str,
        transmission_rate_to_update_flag=transmission_rate_to_update_flag,
        update_type=arguments.assimilation_update_sensor)

viral_test_assimilator = DataAssimilator(
        observations=viral_test_observations,
        errors=[],
        n_assimilation_batches = arguments.assimilation_batches_test,
        transition_rates_to_update_str=transition_rates_to_update_str,
        transmission_rate_to_update_flag=transmission_rate_to_update_flag,
        update_type=arguments.assimilation_update_test,
        joint_cov_noise=arguments.assimilation_regularization)

record_assimilator = DataAssimilator(
        observations=record_observations,
        errors=[],
        n_assimilation_batches=arguments.assimilation_batches_record,
        transition_rates_to_update_str=transition_rates_to_update_str,
        transmission_rate_to_update_flag=transmission_rate_to_update_flag,
        update_type=arguments.assimilation_update_record)

# master equations #############################################################
from _master_eqn_init import (master_eqn_ensemble,
                              ensemble_size,
                              ensemble_ic,
                              transition_rates_ensemble,
                              community_transmission_rate_ensemble,
                              n_forward_steps,
                              n_backward_steps)

# post-processing ##############################################################
from _post_process_init import axes

# inverventions ################################################################
from _intervention_init import (intervention,
                                intervention_frequency,
                                intervention_nodes, 
                                intervention_type) 

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
da_window         = 7.0
prediction_window = 1.0
closure_update_interval = 5.0 # the frequency at which we update the closure
record_assimilation_interval = 1.0 # assimilate H and D data every .. days
test_assimilation_interval  = 1.0 # same for I
sensor_assimilation_interval  = 1.0 # same for I

intervention_start_time = arguments.intervention_start_time
intervention_interval = arguments.intervention_interval
#ints
n_sweeps                    = 1
n_prediction_windows_spin_up = 8
n_prediction_windows        = int(total_time/prediction_window)
steps_per_da_window         = int(da_window/static_contact_interval)
steps_per_prediction_window = int(prediction_window/static_contact_interval)

assert n_prediction_windows_spin_up * prediction_window + prediction_window > da_window
earliest_assimilation_time = (n_prediction_windows_spin_up + 1)* prediction_window - da_window 
assert n_prediction_windows > n_prediction_windows_spin_up

# epidemic storage #############################################################
# Set an upper limit on number of stored contact networks:
max_networks = steps_per_da_window + steps_per_prediction_window 
epidemic_data_storage = StaticIntervalDataSeries(static_contact_interval, max_networks=max_networks)

# storing ######################################################################
master_states_timeseries = EnsembleTimeSeries(ensemble_size,
                                              5 * user_population,
                                              time_span.size)

# intial conditions  ###########################################################
#loaded_data = epidemic_data_storage.get_network_from_start_time(
#        start_time=start_time)
#loaded_kinetic_ic = loaded_data.start_statuses

master_eqn_ensemble.set_states_ensemble(ensemble_ic)
master_eqn_ensemble.set_start_time(start_time)

################################################################################
# master equations + data assimilation computation #############################
################################################################################
# spin-up w/o data assimilation ################################################
current_time = start_time
spin_up_steps = n_prediction_windows_spin_up * steps_per_prediction_window
ensemble_state = ensemble_ic

timer_spin_up = timer()

print_info("Spin-up started")
for j in range(spin_up_steps):
    update_closure_now = modulo_is_close_to_zero(current_time,
                                                 closure_update_interval,
                                                 eps=static_contact_interval)
    if update_closure_now:
        update_closure_flag=True
    else:
        update_closure_flag=False


    walltime_master_eqn = 0.0
    master_eqn_ensemble.reset_walltimes()
    #Run kinetic model
    # run
    
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
    
    kinetic_state = epidemic_simulator.kinetic_model.current_statuses
    
    epidemic_data_storage.save_end_statuses_to_network(
            end_time=current_time+static_contact_interval,
            end_statuses=kinetic_state)

    # store for plotting
    user_state = dict_slice(kinetic_state, user_nodes)
    n_S, n_E, n_I, n_H, n_R, n_D = compartments_count(user_state)
    statuses_sum_trace.append([n_S, n_E, n_I, n_H, n_R, n_D])

    kinetic_states_timeseries.append(kinetic_state)
    
    #Run master eqn mode
    master_states_timeseries.push_back(ensemble_state) # storage

    loaded_data = epidemic_data_storage.get_network_from_start_time(
            start_time=current_time)

    user_network.update_from(loaded_data.contact_network)
    master_eqn_ensemble.set_mean_contact_duration(
            user_network.get_edge_weights())
    timer_master_eqn = timer()
    ensemble_state = master_eqn_ensemble.simulate(
            static_contact_interval,
            min_steps=n_forward_steps,
    closure_flag=update_closure_flag)

    current_time += static_contact_interval

    walltime_master_eqn += timer() - timer_master_eqn

    print_info("eval_closure walltime:", master_eqn_ensemble.get_walltime_eval_closure())
    print_info("master equations walltime:", walltime_master_eqn, end='\n\n')

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

    #intervention if required
    # first get the frequency
    if intervention_frequency == "none":
        intervene_now=False
    elif intervention_frequency == "single":
        intervene_now = (abs(current_time - intervention_start_time) < 0.1*static_contact_interval)
    elif intervention_frequency == "interval":
        if current_time > intervention_start_time - 0.1 * static_contact_interval:
            intervene_now = modulo_is_close_to_zero(current_time,
                                                    intervention_interval,
                                                    eps=static_contact_interval)
    else:
        raise ValueError("unknown 'intervention_frequency', choose from 'none' (default), 'single', or 'interval' ")
    
    
    if intervene_now:
        # now see which nodes have intervention applied
        if intervention_nodes == "all":
            nodes_to_intervene = user_nodes
            print("intervention applied to all {:d} nodes.".format(
                network.get_node_count()))
            
        elif intervention_nodes == "sick":
            nodes_to_intervene = intervention.find_sick(ensemble_state)
            print("intervention applied to sick nodes: {:d}/{:d}".format(
                sick_nodes.size, network.get_node_count()))
        else:
            raise ValueError("unknown 'intervention_nodes', choose from 'all' (default), 'sick'")

        # Apply the the chosen form of intervention
        if intervention_type == "isolate":
            network.isolate(nodes_to_intervene) 

        elif intervention_type == "social_distance":
            λ_min, λ_max = network.get_lambdas()
            λ_max[nodes_to_intervene] = distanced_max_contact_rate
            network.set_lambdas(λ_min,λ_max)

        else:
            raise ValueError("unknown intervention type, choose from 'social_distance' (default), 'isolate' ")


        
print_info("Spin-up ended; elapsed:", timer() - timer_spin_up, end='\n\n')
print_info("Spin-up ended: current time", current_time)

# main loop: backward/forward/data assimilation ################################
# 3 stages per loop:
# 1a) run epidemic for the duration of the prediction window 
# 1b) prediction (no assimilation) forwards steps_per_prediction_window
#    - Save data during this window from [ start , end )
#    - Make observations and store them in the assimilator
# 2) backward assimilation for steps_per_da_window
#    - assimilate first, then run eqn
#    - a final assimilation for the end of window
# 3) forward assimilation for steps_per_da_window
#    - run first, then assimilate
# Repeat from 1)
#
# Then save data after final loop

for k in range(n_prediction_windows_spin_up, n_prediction_windows):
    print_info("Prediction window: {}/{}".format(k+1, n_prediction_windows))
    timer_window = timer()
    walltime_master_eqn = 0.0
    master_eqn_ensemble.reset_walltimes()

    assert are_close(current_time,
                     k * prediction_window,
                     eps=static_contact_interval)
    current_time = k * prediction_window # to avoid build-up of errors

    
    ## 1a) Run epidemic simulator
    ## 1b) forward run w/o data assimilation; prediction
    master_eqn_ensemble.set_start_time(current_time)
    for j in range(steps_per_prediction_window):
        
        update_closure_now = modulo_is_close_to_zero(current_time,
                                                     closure_update_interval,
                                                     eps=static_contact_interval)
        if update_closure_now:
            update_closure_flag=True
        else:
            update_closure_flag=False
        
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
        master_states_timeseries.push_back(ensemble_state)
        
        loaded_data = epidemic_data_storage.get_network_from_start_time(start_time=current_time)

        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(user_network.get_edge_weights())

        # run ensemble forward
        timer_master_eqn = timer()
        ensemble_state = master_eqn_ensemble.simulate(static_contact_interval,
                                                      min_steps=n_forward_steps,
                                                      closure_flag=update_closure_flag)
        walltime_master_eqn += timer() - timer_master_eqn

        current_time += static_contact_interval

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

        
    print_info("Prediction ended: current time:", current_time)

    for i_sweep in range(n_sweeps):
        print_info("Start the DA sweep: {}/{}".format(i_sweep+1, n_sweeps))
        ## 2) backward run with data assimilation
        past_time = current_time # until end of the loop 'current_time' is const
        master_eqn_ensemble.set_start_time(past_time)

        for j in range(steps_per_da_window):
        
            update_closure_now = modulo_is_close_to_zero(past_time,
                                                         closure_update_interval,
                                                         eps=static_contact_interval)
            if update_closure_now:
                update_closure_flag=True
            else:
                update_closure_flag=False
        
            
            loaded_data = epidemic_data_storage.get_network_from_end_time(end_time=past_time)

            # data assimilation
            assimilate_sensor_now = modulo_is_close_to_zero(past_time,
                                                            sensor_assimilation_interval,
                                                            eps=static_contact_interval)
            
            if assimilate_sensor_now:
                (ensemble_state,
                 transition_rates_ensemble,
                 community_transmission_rate_ensemble
                ) = sensor_assimilator.update(
                        ensemble_state,
                        loaded_data.end_statuses,
                        transition_rates_ensemble,
                        community_transmission_rate_ensemble,
                        user_network,
                        past_time)

            assimilate_test_now = modulo_is_close_to_zero(past_time,
                                                       test_assimilation_interval,
                                                       eps=static_contact_interval)

            delay_satisfied = past_time <= (current_time - test_result_delay)

            if assimilate_test_now and delay_satisfied:
                (ensemble_state,
                 transition_rates_ensemble,
                 community_transmission_rate_ensemble
                ) = viral_test_assimilator.update(
                        ensemble_state,
                        loaded_data.end_statuses,
                        transition_rates_ensemble,
                        community_transmission_rate_ensemble,
                        user_network,
                        past_time)

            assimilate_record_now = modulo_is_close_to_zero(past_time,
                                                        record_assimilation_interval,
                                                        eps=static_contact_interval)
            if assimilate_record_now:
                (ensemble_state,
                 transition_rates_ensemble,
                 community_transmission_rate_ensemble
                ) = record_assimilator.update(
                        ensemble_state,
                        loaded_data.end_statuses,
                        transition_rates_ensemble,
                        community_transmission_rate_ensemble,
                        user_network,
                        past_time)

            # update ensemble after data assimilation
            if (assimilate_test_now and delay_satisfied) or (assimilate_record_now):
                master_eqn_ensemble.set_states_ensemble(ensemble_state)
                master_eqn_ensemble.update_ensemble(
                        new_transition_rates=transition_rates_ensemble,
                        new_transmission_rate=community_transmission_rate_ensemble)

            # run ensemble backwards
            user_network.update_from(loaded_data.contact_network)
            master_eqn_ensemble.set_mean_contact_duration(user_network.get_edge_weights())

            timer_master_eqn = timer()
            ensemble_state = master_eqn_ensemble.simulate_backwards(static_contact_interval,
                                                                    min_steps=n_backward_steps,
                                                                    closure_flag=update_closure_flag)

            walltime_master_eqn += timer() - timer_master_eqn

            past_time -= static_contact_interval

        # furthest-in-the-past assimilation (at the peak of the sweep)
        assimilate_sensor_now = modulo_is_close_to_zero(past_time,
                                                        sensor_assimilation_interval,
                                                        eps=static_contact_interval)
            
        if assimilate_sensor_now:
            (ensemble_state,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = sensor_assimilator.update(
                ensemble_state,
                loaded_data.end_statuses,
                transition_rates_ensemble,
                community_transmission_rate_ensemble,
                user_network,
                past_time)

        assimilate_test_now = modulo_is_close_to_zero(past_time,
                                                   test_assimilation_interval,
                                                   eps=static_contact_interval)
        delay_satisfied = past_time <= (current_time - test_result_delay)

        if assimilate_test_now and delay_satisfied:
            (ensemble_state,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = viral_test_assimilator.update(
                ensemble_state,
                loaded_data.start_statuses,
                transition_rates_ensemble,
                community_transmission_rate_ensemble,
                user_network,
                past_time)
            
        assimilate_record_now = modulo_is_close_to_zero(past_time,
                                                    record_assimilation_interval,
                                                    eps=static_contact_interval)
        if assimilate_record_now:
            (ensemble_state,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = record_assimilator.update(
                ensemble_state,
                loaded_data.start_statuses,
                transition_rates_ensemble,
                community_transmission_rate_ensemble,
                user_network,
                past_time)

        # update ensemble after data assimilation
        if (assimilate_test_now and delay_satisfied) or (assimilate_record_now):
            master_eqn_ensemble.set_states_ensemble(ensemble_state)
            master_eqn_ensemble.update_ensemble(
                    new_transition_rates=transition_rates_ensemble,
                    new_transmission_rate=community_transmission_rate_ensemble)

        print_info("Backward assimilation ended; past_time:", past_time)

        ## 3) forward run with data assimilation
        master_eqn_ensemble.set_start_time(past_time)

        for j in range(steps_per_da_window):
            
            update_closure_now = modulo_is_close_to_zero(past_time,
                                                         closure_update_interval,
                                                         eps=static_contact_interval)
            if update_closure_now:
                update_closure_flag=True
            else:
                update_closure_flag=False

            loaded_data = epidemic_data_storage.get_network_from_start_time(start_time=past_time)

            # run ensemble forward
            user_network.update_from(loaded_data.contact_network)
            master_eqn_ensemble.set_mean_contact_duration(user_network.get_edge_weights())

            timer_master_eqn = timer()
            ensemble_state = master_eqn_ensemble.simulate(static_contact_interval,
                                                          min_steps=n_forward_steps,
                                                          closure_flag=update_closure_flag)

            walltime_master_eqn += timer() - timer_master_eqn

            past_time += static_contact_interval

            # data assimilation
            assimilate_sensor_now = modulo_is_close_to_zero(past_time,
                                                            sensor_assimilation_interval,
                                                            eps=static_contact_interval)
            
            if assimilate_sensor_now:
                (ensemble_state,
                 transition_rates_ensemble,
                 community_transmission_rate_ensemble
                ) = sensor_assimilator.update(
                    ensemble_state,
                    loaded_data.end_statuses,
                    transition_rates_ensemble,
                    community_transmission_rate_ensemble,
                    user_network,
                    past_time)

            assimilate_test_now = modulo_is_close_to_zero(past_time,
                                                       test_assimilation_interval,
                                                       eps=static_contact_interval)
            delay_satisfied = past_time <= (current_time - test_result_delay)

            if assimilate_test_now and delay_satisfied:
                (ensemble_state,
                 transition_rates_ensemble,
                 community_transmission_rate_ensemble
                ) = viral_test_assimilator.update(
                        ensemble_state,
                        loaded_data.end_statuses,
                        transition_rates_ensemble,
                        community_transmission_rate_ensemble,
                        user_network,
                        past_time)

            assimilate_record_now = modulo_is_close_to_zero(past_time,
                                                        record_assimilation_interval,
                                                        eps=static_contact_interval)
            if assimilate_record_now:
                (ensemble_state,
                 transition_rates_ensemble,
                 community_transmission_rate_ensemble
                ) = record_assimilator.update(
                        ensemble_state,
                        loaded_data.end_statuses,
                        transition_rates_ensemble,
                        community_transmission_rate_ensemble,
                        user_network,
                        past_time)

            # update ensemble after data assimilation
            if (assimilate_test_now and delay_satisfied) or (assimilate_record_now):
                master_eqn_ensemble.set_states_ensemble(ensemble_state)
                master_eqn_ensemble.update_ensemble(
                        new_transition_rates=transition_rates_ensemble,
                        new_transmission_rate=community_transmission_rate_ensemble)

        print_info("Forward assimilation ended; current_time", current_time)

        # DA should get back to the current time
        assert are_close(past_time, current_time, eps=static_contact_interval)
    
    print_info("Prediction window: {}/{}".format(k+1, n_prediction_windows),
               "ended; elapsed:",
               timer() - timer_window)

    print_info("Prediction window: {}/{};".format(k+1, n_prediction_windows),
               "eval_closure walltime:",
               master_eqn_ensemble.get_walltime_eval_closure())

    print_info("Prediction window: {}/{};".format(k+1, n_prediction_windows),
               "master equations walltime:",
               walltime_master_eqn, end='\n\n')

    start_end_times = [k for k in epidemic_data_storage.static_network_series.keys()]

    first_start_end_time = min(start_end_times, key=get_start_time)
    last_start_end_time = max(start_end_times, key=get_start_time)

    print_info("First network start-end time: ", first_start_end_time.start, first_start_end_time.end)
    print_info("Last network start-end time: ", last_start_end_time.start, last_start_end_time.end)

    #4) Intervention
    # first get the frequency
    if intervention_frequency == "none":
        intervene_now = False
    elif intervention_frequency == "single":
        intervene_now = (abs(current_time - intervention_start_time) < 0.1*static_contact_interval)
    elif intervention_frequency == "interval":
        if current_time > intervention_start_time - 0.1 * static_contact_interval:
            intervene_now = modulo_is_close_to_zero(current_time,
                                                    intervention_interval,
                                                    eps=static_contact_interval)
    else:
        raise ValueError("unknown 'intervention_frequency', choose from 'none' (default), 'single', or 'interval' ")
    
    
    if intervene_now:
        # now see which nodes have intervention applied
        if intervention_nodes == "all":
            nodes_to_intervene = user_nodes
            print("intervention applied to all {:d} nodes".format(
                  network.get_node_count()))
            
        elif intervention_nodes == "sick":
            nodes_to_intervene = intervention.find_sick(ensemble_state)
            print("intervention applied to sick nodes: {:d}/{:d}".format(
                sick_nodes.size, network.get_node_count()))
        else:
            raise ValueError("unknown 'intervention_nodes', choose from 'all' (default), 'sick'")

        # Apply the the chosen form of intervention
        if intervention_type == "isolate":
            network.isolate(nodes_to_intervene) 

        elif intervention_type == "social_distance":
            λ_min, λ_max = network.get_lambdas()
            λ_max[nodes_to_intervene] = distanced_max_contact_rate
            network.set_lambdas(λ_min,λ_max)

        else:
            raise ValueError("unknown intervention type, choose from 'social_distance' (default), 'isolate' ")



## Final storage after last step
master_states_timeseries.push_back(ensemble_state)

print("finished assimilation")
# save & plot ##################################################################

axes = plot_epidemic_data(population, statuses_sum_trace, axes, time_span)

plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic.png'), rasterized=True, dpi=150)


# plot trajectories
axes = plot_ensemble_states(population,
                            master_states_timeseries.container,
                            time_span,
                            axes=axes,
                            xlims=(-0.1, total_time),
                            a_min=0.0)
plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic_and_master_eqn.png'),
            rasterized=True,
            dpi=150)

# save full the data we require:
master_eqns_mean_states = master_states_timeseries.get_mean()
np.save(os.path.join(OUTPUT_PATH, 'master_eqns_mean_states.npy'), master_eqns_mean_states)

kinetic_eqns_statuses = []
for kinetic_state in kinetic_states_timeseries:
    kinetic_eqns_statuses.append(dict_slice(kinetic_state, user_nodes))

np.save(os.path.join(OUTPUT_PATH, 'kinetic_eqns_statuses.npy'), kinetic_eqns_statuses)


