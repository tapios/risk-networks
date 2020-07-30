from epiforecast.utilities import compartments_count, dict_slice

from _constants import start_time, total_time, static_contact_interval
from _network_init import network
from _stochastic_init import (epidemic_simulator,
                              epidemic_data_storage,
                              kinetic_ic)
from _user_network_init import user_nodes
from _utilities import print_start_of, print_end_of, print_info_module


print_start_of(__name__)
################################################################################
print_info_module(__name__,
                  "Running epidemic_simulator for",
                  total_time,
                  "days")

time = start_time          # float
kinetic_state = kinetic_ic # dict { node : compartment }

user_state = dict_slice(kinetic_state, user_nodes)
n_S, n_E, n_I, n_H, n_R, n_D = compartments_count(user_state)
statuses_sum_trace = []
statuses_sum_trace.append([n_S, n_E, n_I, n_H, n_R, n_D])

kinetic_states_timeseries = []
kinetic_states_timeseries.append(kinetic_state) # storing ic

for i in range(int(total_time/static_contact_interval)):
    # run
    network = epidemic_simulator.run(
            stop_time=epidemic_simulator.time + static_contact_interval,
            current_network=network)

    # store for further usage (master equations etc)
    epidemic_data_storage.save_network_by_start_time(
            start_time=time,
            contact_network=network)
    epidemic_data_storage.save_start_statuses_to_network(
            start_time=time,
            start_statuses=kinetic_state)

    time          = epidemic_simulator.time
    kinetic_state = epidemic_simulator.kinetic_model.current_statuses

    epidemic_data_storage.save_end_statuses_to_network(
            end_time=time,
            end_statuses=kinetic_state)

    # store for plotting
    user_state = dict_slice(kinetic_state, user_nodes)
    n_S, n_E, n_I, n_H, n_R, n_D = compartments_count(user_state)
    statuses_sum_trace.append([n_S, n_E, n_I, n_H, n_R, n_D])

    kinetic_states_timeseries.append(kinetic_state)

################################################################################
print_end_of(__name__)

