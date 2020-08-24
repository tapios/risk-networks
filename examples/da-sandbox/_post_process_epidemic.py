from matplotlib import pyplot as plt

from epiforecast.epiplots import plot_epidemic_data
from epiforecast.utilities import compartments_count, dict_slice

from _constants import time_span
from _network_init import population
from _user_network_init import user_nodes
from _post_process_init import axes
from _get_epidemic_data import kinetic_states_timeseries
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
statuses_sum_trace = []

for kinetic_state in kinetic_states_timeseries:
    user_state = dict_slice(kinetic_state, user_nodes)
    n_S, n_E, n_I, n_H, n_R, n_D = compartments_count(user_state)
    statuses_sum_trace.append([n_S, n_E, n_I, n_H, n_R, n_D])

axes = plot_epidemic_data(population, statuses_sum_trace, axes, time_span)

################################################################################
print_end_of(__name__)

