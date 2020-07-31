import os, sys; sys.path.append(os.path.join('..', '..'))

from timeit import default_timer as timer
import numpy as np
from matplotlib import pyplot as plt

from epiforecast.user_base import FullUserGraphBuilder
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.time_series import EnsembleTimeSeries
from epiforecast.risk_simulator_initial_conditions import kinetic_to_master_same_fraction, random_risk_range
from epiforecast.epiplots import plot_roc_curve, plot_ensemble_states
from epiforecast.performance_metrics import TrueNegativeRate, TruePositiveRate, PerformanceTracker
from epiforecast.utilities import dict_slice



################################################################################
# initialization ###############################################################
################################################################################
# arguments parsing ############################################################
import _argparse_init

# constants ####################################################################
from _constants import (static_contact_interval,
                        start_time,
                        end_time,
                        total_time,
                        time_span,
                        OUTPUT_PATH)

# utilities ####################################################################
from _utilities import (print_info,
                        list_of_transition_rates_to_array,
                        modulo_is_close_to_zero,
                        are_close)

# general init #################################################################
import _general_init

# contact network ##############################################################
from _network_init import population

# stochastic model #############################################################
import _stochastic_init

# user network #################################################################
from _user_network_init import user_network, user_nodes, user_population

# observations #################################################################
from _observations_init import (random_infection_test,
                                budgeted_random_infection_test,
                                high_var_infection_test,
                                positive_hospital_records,
                                negative_hospital_records,
                                positive_death_records,
                                negative_death_records)

imperfect_observations = [budgeted_random_infection_test]
perfect_observations   = [positive_hospital_records,
                          negative_hospital_records,
                          positive_death_records,
                          negative_death_records]

I_observation_delay = 0.0

# assimilator ##################################################################
transition_rates_to_update_str   = []
transmission_rate_to_update_flag = False

assimilator_imperfect_observations = DataAssimilator(
        observations=imperfect_observations,
        errors=[],
        transition_rates_to_update_str=transition_rates_to_update_str,
        transmission_rate_to_update_flag=transmission_rate_to_update_flag)

assimilator_perfect_observations = DataAssimilator(
        observations=perfect_observations,
        errors=[],
        n_assimilation_batches = 4,
        transition_rates_to_update_str=transition_rates_to_update_str,
        transmission_rate_to_update_flag=transmission_rate_to_update_flag)

# master equations #############################################################
from _master_eqn_init import (master_eqn_ensemble,
                              ensemble_size,
                              transition_rates_ensemble,
                              community_transmission_rate_ensemble,
                              n_forward_steps,
                              n_backward_steps)

# post-processing ##############################################################
from _post_process_init import axes



################################################################################
# epidemic data ################################################################
################################################################################
from _run_and_store_epidemic import (epidemic_data_storage,
                                     kinetic_states_timeseries)
import _post_process_epidemic

plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic.png'), rasterized=True, dpi=150)



################################################################################
# master equations + data assimilation init ####################################
################################################################################
# constants ####################################################################

da_window         = 7.0
prediction_window = 1.0
HD_assimilation_interval = 1.0 # assimilate H and D data every .. days
I_assimilation_interval  = 1.0 # same for I

n_prediction_windows_spin_up = 12
n_prediction_windows        = int(total_time/prediction_window)
steps_per_da_window         = int(da_window/static_contact_interval)
steps_per_prediction_window = int(prediction_window/static_contact_interval)

assert n_prediction_windows_spin_up * prediction_window + prediction_window > da_window 
assert n_prediction_windows > n_prediction_windows_spin_up

# storing ######################################################################
master_states_timeseries = EnsembleTimeSeries(ensemble_size,
                                              5 * user_population,
                                              time_span.size)

# initial conditions ###########################################################
loaded_data = epidemic_data_storage.get_network_from_start_time(
        start_time=start_time)
loaded_kinetic_ic = loaded_data.start_statuses

ensemble_ic = random_risk_range(population,
                                0.001,
                                0.01,
                                ensemble_size)

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
    master_states_timeseries.push_back(ensemble_state) # storage

    loaded_data = epidemic_data_storage.get_network_from_start_time(
            start_time=current_time)

    user_network.update_from(loaded_data.contact_network)
    master_eqn_ensemble.set_mean_contact_duration(
            user_network.get_edge_weights())

    ensemble_state = master_eqn_ensemble.simulate(
            static_contact_interval,
            min_steps=n_forward_steps)

    current_time += static_contact_interval

print_info("Spin-up ended; elapsed:", timer() - timer_spin_up, end='\n\n')
print_info("Spin-up ended: current time", current_time)

# main loop: backward/forward/data assimilation ################################
# 3 stages per loop: 
# 1) prediction (no assimilation) forwards steps_per_prediction_window
#    - Save data during this window from [ start , end )
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

    assert are_close(current_time,
                     k * prediction_window,
                     eps=static_contact_interval)
    current_time = k * prediction_window # to avoid build-up of errors

    ## 1) forward run w/o data assimilation; prediction
    master_eqn_ensemble.set_start_time(current_time)
    for j in range(steps_per_prediction_window):
        # storage of data first (we do not store end of prediction window)
        master_states_timeseries.push_back(ensemble_state)
        
        loaded_data = epidemic_data_storage.get_network_from_start_time(start_time=current_time)

        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(user_network.get_edge_weights())

        # run ensemble forward
        ensemble_state = master_eqn_ensemble.simulate(static_contact_interval,
                                                      min_steps=n_forward_steps)
        current_time += static_contact_interval

    print_info("Prediction ended: current time:", current_time)

    ## 2) backward run with data assimilation
    past_time = current_time # until end of the loop 'current_time' is const
    master_eqn_ensemble.set_start_time(past_time)

    for j in range(steps_per_da_window):
        loaded_data = epidemic_data_storage.get_network_from_end_time(end_time=past_time)

        # data assimilation
        assimilate_I_now = modulo_is_close_to_zero(past_time,
                                                   I_assimilation_interval,
                                                   eps=static_contact_interval)
        delay_satisfied = past_time <= (current_time - I_observation_delay)

        if assimilate_I_now and delay_satisfied:
            (ensemble_state,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = assimilator_imperfect_observations.update(
                    ensemble_state,
                    loaded_data.end_statuses,
                    transition_rates_ensemble,
                    community_transmission_rate_ensemble,
                    user_network,
                    past_time)

        assimilate_HD_now = modulo_is_close_to_zero(past_time,
                                                    HD_assimilation_interval,
                                                    eps=static_contact_interval)
        if assimilate_HD_now:
            (ensemble_state,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = assimilator_perfect_observations.update(
                    ensemble_state,
                    loaded_data.end_statuses,
                    transition_rates_ensemble,
                    community_transmission_rate_ensemble,
                    user_network,
                    past_time)

        # update ensemble after data assimilation
        if (assimilate_I_now and delay_satisfied) or (assimilate_HD_now):
            master_eqn_ensemble.set_states_ensemble(ensemble_state)
            master_eqn_ensemble.update_ensemble(
                    new_transition_rates=transition_rates_ensemble,
                    new_transmission_rate=community_transmission_rate_ensemble)

        # run ensemble backwards
        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(user_network.get_edge_weights())

        ensemble_state = master_eqn_ensemble.simulate_backwards(static_contact_interval,
                                                                min_steps=n_backward_steps)
        past_time -= static_contact_interval

    # furthest-in-the-past assimilation (at the peak of the sweep)
    assimilate_I_now = modulo_is_close_to_zero(past_time,
                                               I_assimilation_interval,
                                               eps=static_contact_interval)
    delay_satisfied = past_time <= (current_time - I_observation_delay)

    if assimilate_I_now and delay_satisfied:
        (ensemble_state,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
        ) = assimilator_imperfect_observations.update(
            ensemble_state,
            loaded_data.start_statuses,
            transition_rates_ensemble,
            community_transmission_rate_ensemble,
            user_network,
            past_time)
        
        assimilate_HD_now = modulo_is_close_to_zero(past_time,
                                                    HD_assimilation_interval,
                                                    eps=static_contact_interval)
    if assimilate_HD_now:
        (ensemble_state,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
        ) = assimilator_perfect_observations.update(
            ensemble_state,
            loaded_data.start_statuses,
            transition_rates_ensemble,
            community_transmission_rate_ensemble,
            user_network,
            past_time)

    # update ensemble after data assimilation
    if (assimilate_I_now and delay_satisfied) or (assimilate_HD_now):
        master_eqn_ensemble.set_states_ensemble(ensemble_state)
        master_eqn_ensemble.update_ensemble(
            new_transition_rates=transition_rates_ensemble,
            new_transmission_rate=community_transmission_rate_ensemble)

    print_info("Backward assimilation ended; current time:", past_time)

    ## 3) forward run with data assimilation
    master_eqn_ensemble.set_start_time(past_time)

    for j in range(steps_per_da_window):
        loaded_data = epidemic_data_storage.get_network_from_start_time(start_time=past_time)

        # run ensemble forward
        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(user_network.get_edge_weights())

        ensemble_state = master_eqn_ensemble.simulate(static_contact_interval,
                                                      min_steps=n_forward_steps)
        past_time += static_contact_interval

        # data assimilation
        assimilate_I_now = modulo_is_close_to_zero(past_time,
                                                   I_assimilation_interval,
                                                   eps=static_contact_interval)
        delay_satisfied = past_time <= (current_time - I_observation_delay)

        if assimilate_I_now and delay_satisfied:
            (ensemble_state,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = assimilator_imperfect_observations.update(
                    ensemble_state,
                    loaded_data.end_statuses,
                    transition_rates_ensemble,
                    community_transmission_rate_ensemble,
                    user_network,
                    past_time)

        assimilate_HD_now = modulo_is_close_to_zero(past_time,
                                                    HD_assimilation_interval,
                                                    eps=static_contact_interval)
        if assimilate_HD_now:
            (ensemble_state,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = assimilator_perfect_observations.update(
                    ensemble_state,
                    loaded_data.end_statuses,
                    transition_rates_ensemble,
                    community_transmission_rate_ensemble,
                    user_network,
                    past_time)

        # update ensemble after data assimilation
        if (assimilate_I_now and delay_satisfied) or (assimilate_HD_now):
            master_eqn_ensemble.set_states_ensemble(ensemble_state)
            master_eqn_ensemble.update_ensemble(
                    new_transition_rates=transition_rates_ensemble,
                    new_transmission_rate=community_transmission_rate_ensemble)

    print_info("Forward assimilation ended; current time", past_time)

    # DA should get back to the current time
    assert are_close(past_time, current_time, eps=static_contact_interval)

    print_info("Prediction window: {}/{}".format(k+1, n_prediction_windows),
               "ended; elapsed:", timer() - timer_window, end='\n\n')

## Final storage after last step
master_states_timeseries.push_back(ensemble_state)


# save & plot ##################################################################
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


