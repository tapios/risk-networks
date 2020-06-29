import os, sys; sys.path.append(os.path.join("../.."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

from numba import set_num_threads

set_num_threads(1)

from epiforecast.populations import TransitionRates
from epiforecast.samplers import AgeDependentConstant
from epiforecast.contact_network import ContactNetwork
from epiforecast.scenarios import random_epidemic
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.utilities import seed_three_random_states


#
# Set random seeds for reproducibility
#

seed = 942395
seed_three_random_states(seed)

#
# Load an example network
#

edges_filename = os.path.join('../..', 'data', 'networks',
                              'edge_list_SBM_1e3_nobeds.txt')
groups_filename = os.path.join('../..', 'data', 'networks',
                               'node_groups_SBM_1e3_nobeds.json')

network = ContactNetwork.from_files(edges_filename, groups_filename)
population = network.get_node_count()
populace = network.get_nodes()


minute = 1 / 60 / 24
hour = 60 * minute

#
# Clinical parameters of an age-distributed population
#
age_distribution =[0.21, 0.4, 0.25, 0.08, 0.06]
health_workers_subset = [1, 2] # which age groups to draw from for h-workers
assert sum(age_distribution) == 1.0
network.draw_and_set_age_groups(age_distribution, health_workers_subset)

# We process the clinical data to determine transition rates between each epidemiological state,

latent_periods = 3.7
community_infection_periods = 3.2
hospital_infection_periods = 5.0
hospitalization_fraction = AgeDependentConstant([0.002,  0.01,   0.04, 0.076,  0.16])
community_mortality_fraction = AgeDependentConstant([ 1e-4,  1e-3,  0.001,  0.07,  0.015])
hospital_mortality_fraction = AgeDependentConstant([0.019, 0.073,  0.193, 0.327, 0.512])

transition_rates = TransitionRates(population = network.get_node_count(),
                                   lp_sampler = latent_periods,
                                  cip_sampler = community_infection_periods,
                                  hip_sampler = hospital_infection_periods,
                                   hf_sampler = hospitalization_fraction,
                                  cmf_sampler = community_mortality_fraction,
                                  hmf_sampler = hospital_mortality_fraction,
                    distributional_parameters = network.get_age_groups()
)

transition_rates.calculate_from_clinical() 
network.set_transition_rates_for_kinetic_model(transition_rates)

community_transmission_rate = 12.0



#
# Set up the epidemic simulator and health service
#
health_service = HealthService(original_contact_network = network,
                               health_workers = network.get_health_workers())

mean_contact_lifetime=0.5*minute
static_contact_interval = 3 * hour
hospital_transmission_reduction = 0.1

min_contact_rate = 2
max_contact_rate = 22

network.set_lambdas(min_contact_rate,max_contact_rate)

epidemic_simulator = EpidemicSimulator(
                 contact_network = network,
     community_transmission_rate = community_transmission_rate,
 hospital_transmission_reduction = hospital_transmission_reduction,
         static_contact_interval = static_contact_interval,
           mean_contact_lifetime = mean_contact_lifetime,
              day_inception_rate = max_contact_rate,
            night_inception_rate = min_contact_rate,
                  health_service = health_service
                                      )
