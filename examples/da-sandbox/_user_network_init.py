import numpy as np

from epiforecast.user_base import (FullUserGraphBuilder,
                                   FractionalUserGraphBuilder,
                                   ContiguousUserGraphBuilder,
                                   contiguous_indicators)

from _argparse_init import arguments
from _network_init import network
from _utilities import print_start_of, print_end_of, print_warning_module


print_start_of(__name__)
################################################################################
user_fraction = arguments.user_network_user_fraction

if user_fraction >= 1.0:
    user_network = network.build_user_network_using(FullUserGraphBuilder())
    exterior_neighbors = None
else:
    user_network_type = arguments.user_network_type #"neighbor", or "random"
    if user_network_type == "neighbor":
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
        
    elif user_network_type == "random":
        user_network =  network.build_user_network_using(FractionalUserGraphBuilder(user_fraction))
 
    (interior_nodes, 
     boundary_nodes,
     mean_neighbors_exterior,
     edge_indicator_list,
     node_indicator_list,
     exterior_neighbors) = contiguous_indicators(network.graph,user_network.graph)
    print(interior_nodes, boundary_nodes, mean_neighbors_exterior)

    # if we wish to weight the boundary of the user network
    print("number of weights > 0", sum(exterior_neighbors>0.0))
    print("number of weights > 2", sum(exterior_neighbors>2.0))
    print("number of weights > 5", sum(exterior_neighbors>5.0))

    
user_nodes = user_network.get_nodes()
user_population = user_network.get_node_count()

print("created user network with population", user_population)


################################################################################
print_end_of(__name__)

