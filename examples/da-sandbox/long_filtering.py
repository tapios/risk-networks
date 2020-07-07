import os, sys; sys.path.append(os.path.join('..', '..'))

import pickle
import numpy as np
from matplotlib import pyplot as plt

from epiforecast.user_base import FullUserGraphBuilder
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.time_series import EnsembleTimeSeries
from epiforecast.risk_simulator_initial_conditions import deterministic_risk
from epiforecast.epiplots import plot_ensemble_states



################################################################################
# initialization ###############################################################
################################################################################
# constants ####################################################################
from _constants import (static_contact_interval,
                        start_time,
                        end_time,
                        total_time,
                        time_span,
                        OUTPUT_PATH)

# utilities ####################################################################
from _utilities import list_of_transition_rates_to_array

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
                                positive_hospital_records,
                                negative_hospital_records,
                                positive_death_records,
                                negative_death_records)

imperfect_observations = [random_infection_test]
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

pickle.dump(statuses_all,
            open(os.path.join(OUTPUT_PATH, 'statuses_all.pkl'), 'wb'))
plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic.png'), rasterized=True, dpi=150)



################################################################################
# master equations + data assimilation #########################################
################################################################################
# constants ####################################################################
n_intervals_skipped  = 2
backward_da_interval = 1.0
forward_da_interval  = 1.0
prediction_interval  = 1.0

n_intervals = int(total_time/backward_da_interval)
steps_per_backward_interval  = int(backward_da_interval/static_contact_interval)
steps_per_forward_interval   = int(forward_da_interval/static_contact_interval)
steps_per_prediction_interval= int(prediction_interval/static_contact_interval)

# storing ######################################################################
states_trace_ensemble = np.zeros([ensemble_size,
                                  5 * user_population,
                                  time_span.size])

n_parameters= transition_rates_ensemble[0].get_clinical_parameters_total_count()
n_β         = 1
rates_timeseries = EnsembleTimeSeries(ensemble_size, n_parameters, n_intervals)
β_timeseries     = EnsembleTimeSeries(ensemble_size,          n_β, n_intervals)

# initialization ###############################################################
loaded_data = epidemic_data_storage.get_network_from_start_time(
        start_time=start_time)
loaded_kinetic_ic = loaded_data.start_statuses

# TODO 'deterministic_risk' should take in 'user_nodes';
# but conceptually it should be 'user_*' something, not 'network_*'
# (it used to be 'population')
ensemble_ic = deterministic_risk(user_population,
                                 loaded_kinetic_ic,
                                 ensemble_size=ensemble_size)

master_eqn_ensemble.set_states_ensemble(ensemble_ic)
master_eqn_ensemble.set_start_time(start_time)

# spin-up w/o data assimilation ################################################
current_time = start_time
spin_up_steps = n_intervals_skipped * steps_per_backward_interval

for j in range(spin_up_steps):
    loaded_data = epidemic_data_storage.get_network_from_start_time(
            start_time=current_time)

    # XXX please check: the following three lines changed
    user_network.update_from(loaded_data.contact_network)
    master_eqn_ensemble.set_mean_contact_duration(
            user_network.get_edge_weights())

    ensemble_state = master_eqn_ensemble.simulate(
            static_contact_interval,
            n_steps=n_forward_steps)

    master_eqn_ensemble.set_states_ensemble(ensemble_state)

    current_time += static_contact_interval

    # storing
    states_trace_ensemble[:,:,j] = ensemble_state
    if j % steps_per_backward_interval == 0:
        rates_timeseries.push_back(
                list_of_transition_rates_to_array(transition_rates_ensemble))
        β_timeseries.push_back(community_transmission_rate_ensemble)


# main loop: backward/forward/data assimilation ################################
for k in range(n_intervals_skipped, n_intervals):
    current_time = k * backward_da_interval

    # backward run with data assimilation
    master_eqn_ensemble.set_start_time(current_time)

    for j in range(steps_per_backward_interval):
        loaded_data = epidemic_data_storage.get_network_from_end_time(
                end_time=current_time)

        # XXX please check: the following three lines changed
        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(
                user_network.get_edge_weights())

        # run ensemble backwards
        ensemble_state = master_eqn_ensemble.simulate_backwards(
                static_contact_interval,
                n_steps=n_backward_steps)

        current_time -= static_contact_interval
        # data assimilation
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


    # store parameters
    rates_timeseries.push_back(
            list_of_transition_rates_to_array(transition_rates_ensemble))
    β_timeseries.push_back(community_transmission_rate_ensemble)

    # forward run with data assimilation
    master_eqn_ensemble.set_start_time(current_time)

    for j in range(steps_per_forward_interval):
        loaded_data = epidemic_data_storage.get_network_from_start_time(
                start_time=current_time)

        # XXX please check: the following three lines changed
        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(
                user_network.get_edge_weights())

        # run ensemble forward
        ensemble_state = master_eqn_ensemble.simulate(
                static_contact_interval,
                n_steps=n_forward_steps)

        current_time += static_contact_interval

        # data assimilation
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

    for j in range(steps_per_prediction_interval):
        loaded_data = epidemic_data_storage.get_network_from_start_time(
                start_time=current_time)

        # XXX please check: the following three lines changed
        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(
                user_network.get_edge_weights())

        # run ensemble forward
        ensemble_state = master_eqn_ensemble.simulate(
                static_contact_interval,
                n_steps=n_forward_steps)

        master_eqn_ensemble.set_states_ensemble(ensemble_state)

        current_time += static_contact_interval
        current_step = k * steps_per_backward_interval + j
        states_trace_ensemble[:,:,current_step] = ensemble_state


# save & plot ##################################################################
pickle.dump(states_trace_ensemble,
            open(os.path.join(OUTPUT_PATH, 'states_trace_ensemble.pkl'), 'wb'))
pickle.dump(time_span,
            open(os.path.join(OUTPUT_PATH, 'time_span.pkl'), 'wb'))

# plot trajectories
axes = plot_ensemble_states(states_trace_ensemble,
                            time_span,
                            axes=axes,
                            xlims=(-0.1, total_time),
                            a_min=0.0)
plt.savefig(os.path.join(OUTPUT_PATH, 'epidemic_and_master_eqn.png'),
            rasterized=True,
            dpi=150)

# plot parameters
fig2, axes2 = plt.subplots(1, 1, figsize = (16, 4))

rates_timeseries_mean = rates_timeseries.get_mean()
β_timeseries_mean     = β_timeseries.get_mean()
da_time_span = np.linspace(start_time, end_time, n_intervals)

for name in transition_rates_to_update_str:
    clinical_indices = (
            transition_rates_ensemble[0].get_clinical_parameter_indices(name))
    for i in clinical_indices:
        plt.plot(da_time_span, rates_timeseries_mean[i])

plt.plot(da_time_span, np.squeeze(β_timeseries_mean))

plt.savefig(os.path.join(OUTPUT_PATH, 'model_parameters.png'),
            rasterized=True,
            dpi=150)


