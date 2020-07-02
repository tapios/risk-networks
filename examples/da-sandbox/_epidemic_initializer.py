import os, sys; sys.path.append(os.path.join("../.."))

from timeit import default_timer as timer

import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from numba import set_num_threads

from epiforecast.populations import TransitionRates
from epiforecast.contact_network import ContactNetwork
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
from epiforecast.utilities import seed_three_random_states


################################################################################
# constants ####################################################################
################################################################################
from _constants import *

################################################################################
# initialization ###############################################################
################################################################################
# numba
set_num_threads(1)

# Set random seeds for reproducibility
seed = 942395
seed_three_random_states(seed)

# contact network ##############################################################
edges_filename = os.path.join(NETWORKS_PATH, 'edge_list_SBM_1e3_nobeds.txt')
groups_filename = os.path.join(NETWORKS_PATH, 'node_groups_SBM_1e3_nobeds.json')

network = ContactNetwork.from_files(edges_filename, groups_filename)
network.draw_and_set_age_groups(age_distribution, health_workers_subset)
network.set_lambdas(min_contact_rate, max_contact_rate)

population = network.get_node_count()
populace = network.get_nodes()

# stochastic model #############################################################
# transition rates a.k.a. independent rates (σ, γ etc.)
# constructor takes clinical parameter samplers which are then used to draw real
# clinical parameters, and those are used to calculate transition rates
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

health_service = HealthService(original_contact_network = network,
                               health_workers = network.get_health_workers())

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

epidemic_data_storage = StaticIntervalDataSeries(static_contact_interval)


