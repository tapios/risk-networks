from epiforecast.user_base import FullUserGraphBuilder

from _network_init import network
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
user_network = network.build_user_network_using(FullUserGraphBuilder())

user_nodes = user_network.get_nodes()
user_population = user_network.get_node_count()

################################################################################
print_end_of(__name__)

