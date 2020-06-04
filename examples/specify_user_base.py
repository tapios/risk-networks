import os, sys; sys.path.append(os.path.join(".."))

import networkx as nx
import numpy as np

from epiforecast.user_base import FullUserBase, FractionalUserBase, ContiguousUserBase, contiguous_indicators

np.random.seed(123)

contact_network = nx.watts_strogatz_graph(100000, 12, 0.1, 1)
population = len(contact_network)

# create a full user base
full_user_base=FullUserBase(contact_network)
print("User base: Full")
print("number of nodes", len(full_user_base.contact_network.nodes))
print("number of edges", len(full_user_base.contact_network.edges))

# create a user base from a random fraction of the population
user_fraction = 0.1
fractional_user_base = FractionalUserBase(contact_network,user_fraction)
print("")
print("User base: ", user_fraction, " fraction of nodes, randomly chosen")
print("number of nodes", len(fractional_user_base.contact_network.nodes))
print("number of edges", len(fractional_user_base.contact_network.edges))

interior,boundary,mean_exterior_neighbors = contiguous_indicators(contact_network,fractional_user_base.contact_network)
print("number of interior nodes:", interior)
print("number of boundary nodes", boundary)
print("average exterior neighbours of boundary node", mean_exterior_neighbors)

# create a user base from a Contiguous region about a random seed user (or a specified one)
neighbor_user_base = ContiguousUserBase(contact_network,user_fraction, method="neighbor", seed_user=None)
print("")
print("User base:", user_fraction, " fraction of nodes, chosen using neighbor method")
print("number of nodes", len(neighbor_user_base.contact_network.nodes))
print("number of edges", len(neighbor_user_base.contact_network.edges))

interior,boundary,mean_exterior_neighbors = contiguous_indicators(contact_network,neighbor_user_base.contact_network)
print("number of interior nodes:", interior)
print("number of boundary nodes", boundary)
print("average exterior neighbours of boundary node", mean_exterior_neighbors)


# create a user base from a Contiguous region about a random seed user (or a specified one)
clique_user_base = ContiguousUserBase(contact_network,user_fraction, method="clique", seed_user=None)
print("")
print("User base:", user_fraction, " fraction of nodes, chosen using clique method")
print("number of nodes", len(clique_user_base.contact_network.nodes))
print("number of edges", len(clique_user_base.contact_network.edges))

interior,boundary,mean_exterior_neighbors = contiguous_indicators(contact_network,clique_user_base.contact_network)
print("number of interior nodes:", interior)
print("number of boundary nodes", boundary)
print("average exterior neighbours of boundary node", mean_exterior_neighbors)


