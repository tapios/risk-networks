import numpy as np
import os, sys; sys.path.append(os.path.join('..', '..'))
import matplotlib
import matplotlib.pyplot as plt

from epiforecast.epiplots import plot_ensemble_averaged_clinical_parameters

from _user_network_init import user_network
age_category_of_users = user_network.get_age_groups()

from _constants import (age_indep_transition_rates_true,
                        age_dep_transition_rates_true)

from _constants import time_span

OUTPUT_PATH = 'output/CASE_NAME'
network_transition_rates_timeseries = np.load(os.path.join(OUTPUT_PATH, 
                                              'network_mean_transition_rates.npy'))

plot_ensemble_averaged_clinical_parameters(
    np.swapaxes(network_transition_rates_timeseries, 0, 1),
    time_span,
    age_category_of_users,
    age_indep_rates_true = age_indep_transition_rates_true,
    age_dep_rates_true = age_dep_transition_rates_true,
    a_min=0.0,
    output_path=OUTPUT_PATH,
    output_name='network')
