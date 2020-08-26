import numpy as np

from epiforecast.user_base import (FullUserGraphBuilder,
                                   ContiguousUserGraphBuilder)

from _argparse_init import arguments
from _network_init import network
from _utilities import print_start_of, print_end_of, print_warning_module


print_start_of(__name__)
################################################################################
user_fraction = arguments.user_network_user_fraction

if user_fraction >= 1.0:
    user_network = network.build_user_network_using(FullUserGraphBuilder())
else:
    seed_user = arguments.user_network_seed_user
    n_nodes = network.get_node_count()
    if seed_user > n_nodes:
        print_warning_module(
                __name__,
                "specified seed user is greater than node count: "
                + "seed_user = {}".format(seed_user)
                + ", n_nodes = {}".format(n_nodes))
        print_warning_module(
                __name__,
                "this defaults to using a random seed user")
        seed_user = None

    user_network = network.build_user_network_using(
            ContiguousUserGraphBuilder(user_fraction, seed_user=seed_user))

user_nodes = user_network.get_nodes()
user_population = user_network.get_node_count()

################################################################################
print_end_of(__name__)

