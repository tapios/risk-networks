from matplotlib import pyplot as plt

from epiforecast.epiplots import plot_epidemic_data

from _constants import time_span
from _stochastic_init import epidemic_simulator
from _run_and_store_epidemic import statuses_sum_trace
from _post_process_init import axes
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
axes = plot_epidemic_data(kinetic_model = epidemic_simulator.kinetic_model,
                          statuses_list = statuses_sum_trace,
                                   axes = axes,
                             plot_times = time_span)

################################################################################
print_end_of(__name__)

