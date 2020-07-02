import os

from epiforecast.contact_network import ContactNetwork

from _constants import (NETWORKS_PATH, age_distribution, health_workers_subset,
                        min_contact_rate, max_contact_rate)
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
edges_filename = os.path.join(NETWORKS_PATH, 'edge_list_SBM_1e3_nobeds.txt')
groups_filename = os.path.join(NETWORKS_PATH, 'node_groups_SBM_1e3_nobeds.json')

network = ContactNetwork.from_files(edges_filename, groups_filename)
network.draw_and_set_age_groups(age_distribution, health_workers_subset)
network.set_lambdas(min_contact_rate, max_contact_rate)

population = network.get_node_count()
populace = network.get_nodes()

################################################################################
print_end_of(__name__)

