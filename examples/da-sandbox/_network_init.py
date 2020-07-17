import os

from epiforecast.contact_network import ContactNetwork

from _argparse_init import arguments
from _constants import (NETWORKS_PATH, age_distribution, health_workers_subset,
                        min_contact_rate, max_contact_rate)
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
edges_filename  = ('edge_list_SBM_'
                   + arguments.network_node_count
                   + '_nobeds.txt')
groups_filename = ('node_groups_SBM_'
                   + arguments.network_node_count
                   + '_nobeds.json')
edges_path  = os.path.join(NETWORKS_PATH, edges_filename)
groups_path = os.path.join(NETWORKS_PATH, groups_filename)

network = ContactNetwork.from_files(edges_path, groups_path)
network.draw_and_set_age_groups(age_distribution, health_workers_subset)
network.set_lambdas(min_contact_rate, max_contact_rate)

population = network.get_node_count()
populace = network.get_nodes()

################################################################################
print_end_of(__name__)

