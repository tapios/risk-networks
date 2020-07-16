from matplotlib import pyplot as plt

from epiforecast.epiplots import plot_epidemic_data

from _constants import time_span
from _network_init import population
from _post_process_init import axes
from _run_and_store_epidemic import statuses_sum_trace
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
axes = plot_epidemic_data(population, statuses_sum_trace, axes, time_span)

################################################################################
print_end_of(__name__)

