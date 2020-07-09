import os, sys; sys.path.append(os.path.join('..', '..'))

import numpy as np
from matplotlib import pyplot as plt

from epiforecast.user_base import FullUserGraphBuilder
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.time_series import EnsembleTimeSeries
from epiforecast.risk_simulator_initial_conditions import deterministic_risk
from epiforecast.epiplots import plot_ensemble_states
from epiforecast.performance_metrics import ModelAccuracy, PerformanceTracker
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
                        modulo_is_close_to_zero)

# general init #################################################################
import _general_init

# contact network ##############################################################
from _network_init import population

# stochastic model #############################################################
import _stochastic_init

# user network #################################################################
from _user_network_init import user_network, user_nodes, user_population

# observations #################################################################
from _observations_init import (high_var_infection_test,
                                positive_hospital_records,
                                negative_hospital_records,
                                positive_death_records,
                                negative_death_records)

imperfect_observations = [high_var_infection_test]
perfect_observations   = [positive_hospital_records,
                          negative_hospital_records,
                          positive_death_records,
                          negative_death_records]

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
                                     statuses_all)
import _post_process_epidemic

plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic.png'), rasterized=True, dpi=150)



################################################################################
# master equations + data assimilation init ####################################
################################################################################
# constants ####################################################################
n_prediction_windows_skipped = 10 # for spin-up period

da_window         = 7.0
prediction_window = 1.0
HD_assimilation_interval = 1.0 # assimilate H and D data every .. days
I_assimilation_interval  = static_contact_interval # same for I

n_prediction_windows        = int(total_time/prediction_window)
steps_per_da_window         = int(da_window/static_contact_interval)
steps_per_prediction_window = int(prediction_window/static_contact_interval)

assert n_prediction_windows_skipped * prediction_window > da_window
assert n_prediction_windows > n_prediction_windows_skipped

# storing ######################################################################
master_states = EnsembleTimeSeries(ensemble_size,
                                   5 * user_population,
                                   time_span.size)

# initial conditions ###########################################################
loaded_data = epidemic_data_storage.get_network_from_start_time(
        start_time=start_time)
loaded_kinetic_ic = loaded_data.start_statuses

ensemble_ic = deterministic_risk(user_nodes,
                                 loaded_kinetic_ic,
                                 ensemble_size)

master_eqn_ensemble.set_states_ensemble(ensemble_ic)
master_eqn_ensemble.set_start_time(start_time)



################################################################################
# master equations + data assimilation computation #############################
################################################################################
# spin-up w/o data assimilation ################################################
current_time = start_time
spin_up_steps = n_prediction_windows_skipped * steps_per_prediction_window

print_info("Spin-up started")
for j in range(spin_up_steps):
    loaded_data = epidemic_data_storage.get_network_from_start_time(
            start_time=current_time)

    user_network.update_from(loaded_data.contact_network)
    master_eqn_ensemble.set_mean_contact_duration(
            user_network.get_edge_weights())

    ensemble_state = master_eqn_ensemble.simulate(
            static_contact_interval,
            n_steps=n_forward_steps)

    master_eqn_ensemble.set_states_ensemble(ensemble_state)

    current_time += static_contact_interval

    # storing
    master_states.push_back(ensemble_state)

print_info("Spin-up ended")

# main loop: backward/forward/data assimilation ################################
for k in range(n_prediction_windows_skipped, n_prediction_windows):
    print_info("Prediction window: {}/{}".format(k+1, n_prediction_windows))
    current_time = k * prediction_window

    # backward run with data assimilation
    master_eqn_ensemble.set_start_time(current_time)

    for j in range(steps_per_da_window):
        loaded_data = epidemic_data_storage.get_network_from_end_time(
                end_time=current_time)

        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(
                user_network.get_edge_weights())

        # run ensemble backwards
        ensemble_state = master_eqn_ensemble.simulate_backwards(
                static_contact_interval,
                n_steps=n_backward_steps)

        current_time -= static_contact_interval

        # data assimilation
        if modulo_is_close_to_zero(current_time,
                                   I_assimilation_interval,
                                   eps=static_contact_interval):
            (ensemble_state,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = assimilator_imperfect_observations.update(
                    ensemble_state,
                    loaded_data.start_statuses,
                    transition_rates_ensemble,
                    community_transmission_rate_ensemble,
                    user_nodes,
                    current_time)

        if modulo_is_close_to_zero(current_time,
                                   HD_assimilation_interval,
                                   eps=static_contact_interval):
            (ensemble_state,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = assimilator_perfect_observations.update(
                    ensemble_state,
                    loaded_data.start_statuses,
                    transition_rates_ensemble,
                    community_transmission_rate_ensemble,
                    user_nodes,
                    current_time)

        # update ensemble after data assimilation
        master_eqn_ensemble.set_states_ensemble(ensemble_state)
        master_eqn_ensemble.update_ensemble(
                new_transition_rates=transition_rates_ensemble,
                new_transmission_rate=community_transmission_rate_ensemble)


    # forward run with data assimilation
    master_eqn_ensemble.set_start_time(current_time)

    for j in range(steps_per_da_window):
        loaded_data = epidemic_data_storage.get_network_from_start_time(
                start_time=current_time)

        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(
                user_network.get_edge_weights())

        # run ensemble forward
        ensemble_state = master_eqn_ensemble.simulate(
                static_contact_interval,
                n_steps=n_forward_steps)

        current_time += static_contact_interval

        # data assimilation
        if modulo_is_close_to_zero(current_time,
                                   I_assimilation_interval,
                                   eps=static_contact_interval):
            (ensemble_state,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = assimilator_imperfect_observations.update(
                    ensemble_state,
                    loaded_data.end_statuses,
                    transition_rates_ensemble,
                    community_transmission_rate_ensemble,
                    user_nodes,
                    current_time)

        if modulo_is_close_to_zero(current_time,
                                   HD_assimilation_interval,
                                   eps=static_contact_interval):
            (ensemble_state,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = assimilator_perfect_observations.update(
                    ensemble_state,
                    loaded_data.end_statuses,
                    transition_rates_ensemble,
                    community_transmission_rate_ensemble,
                    user_nodes,
                    current_time)

        # update ensemble after data assimilation
        master_eqn_ensemble.set_states_ensemble(ensemble_state)
        master_eqn_ensemble.update_ensemble(
                new_transition_rates=transition_rates_ensemble,
                new_transmission_rate=community_transmission_rate_ensemble)


    # forward run w/o data assimilation; prediction
    master_eqn_ensemble.set_start_time(current_time)

    for j in range(steps_per_prediction_window):
        loaded_data = epidemic_data_storage.get_network_from_start_time(
                start_time=current_time)

        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(
                user_network.get_edge_weights())

        # run ensemble forward
        ensemble_state = master_eqn_ensemble.simulate(
                static_contact_interval,
                n_steps=n_forward_steps)

        master_eqn_ensemble.set_states_ensemble(ensemble_state)

        current_time += static_contact_interval
        master_states.push_back(ensemble_state)


# save & plot ##################################################################
accuracy_tracker = PerformanceTracker(metrics=[ModelAccuracy()])
for j, kinetic_state in enumerate(statuses_all):
    user_kinetic_state = dict_slice(kinetic_state, user_nodes)
    accuracy_tracker.update(user_kinetic_state, master_states.get_snapshot(j))

np.save(os.path.join(OUTPUT_PATH, 'accuracy_track.npy'),
        accuracy_tracker.performance_track)
np.save(os.path.join(OUTPUT_PATH, 'time_span.npy'),
        time_span)

# plot trajectories
axes = plot_ensemble_states(population,
                            master_states.container,
                            time_span,
                            axes=axes,
                            xlims=(-0.1, total_time),
                            a_min=0.0)
plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic_and_master_eqn.png'),
            rasterized=True,
            dpi=150)

# plot accuracy
_, axes_acc = plt.subplots()
axes_acc.plot(time_span, accuracy_tracker.performance_track)

plt.savefig(os.path.join(OUTPUT_PATH, 'accuracy.png'),
            rasterized=True,
            dpi=150)


