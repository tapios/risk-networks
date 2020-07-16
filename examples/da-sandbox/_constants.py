import os
from datetime import datetime
import numpy as np

from epiforecast.samplers import AgeDependentConstant

from _argparse_init import arguments
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
# paths, flags etc #############################################################
NETWORKS_PATH = os.path.join('..', '..', 'data', 'networks')
SIMULATION_PATH = os.path.join('..', '..', 'data', 'simulation_data')
FIGURES_PATH = os.path.join('..', '..', 'figs')

if len(arguments.constants_output_path) > 0:
    OUTPUT_PATH = arguments.constants_output_path
else:
    OUTPUT_PATH = os.path.join('output',
                               datetime.now().strftime('%y%m%d-%H-%M'))

# time & intervals #############################################################
minute = 1 / 60 / 24
hour = 60 * minute
day = 1.0

static_contact_interval = 3 * hour
mean_contact_lifetime = 0.5 * minute

start_time  = 0.0   # the ultimate start time, i.e. when the simulation starts
end_time    = 45.0  # the ultimate end time
total_time  = end_time - start_time
total_steps = int(total_time/static_contact_interval)

time_span = np.linspace(start_time, end_time, total_steps, endpoint=False)

# model parameters #############################################################
# 5 age groups (0-17, 18-44, 45-64, 65-74, >=75) and their respective rates
age_distribution = [0.207, 0.400, 0.245, 0.083, 0.065]
health_workers_subset = [1, 2] # which age groups to draw from for h-workers
age_dep_h      = [0.002   ,  0.010  ,  0.040,  0.076,  0.160]
age_dep_d      = [0.000001,  0.00001,  0.001,  0.007,  0.015]
age_dep_dprime = [0.019   ,  0.073  ,  0.193,  0.327,  0.512]

assert sum(age_distribution) == 1.0

min_contact_rate = 2
max_contact_rate = 22

latent_periods               = 3.7 # == 1/σ
community_infection_periods  = 3.2 # == 1/γ
hospital_infection_periods   = 5.0 # == 1/γ_prime
hospitalization_fraction     = AgeDependentConstant(age_dep_h)
community_mortality_fraction = AgeDependentConstant(age_dep_d)
hospital_mortality_fraction  = AgeDependentConstant(age_dep_dprime)

community_transmission_rate = 12.0    # == β
hospital_transmission_reduction = 0.1 # == α

################################################################################
print_end_of(__name__)

