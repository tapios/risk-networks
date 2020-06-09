import os, sys; sys.path.append(os.path.join(".."))

import networkx as nx
import numpy as np

from epiforecast.user_base import FullUserBase, FractionalUserBase, ContiguousUserBase
from epiforecase.scenarios import load_edges


np.random.seed(123)

#Create network from nx function:
contact_network = nx.watts_strogatz_graph(100000, 12, 0.1, 1)
population = len(contact_network)


# Or create from file:
edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e4_nobeds.txt')) 

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)


# create a full user base
full_user_base=FullUserBase(contact_network)
print(len(full_user_base.contact_network.nodes))

# create a user base from a random fraction of the population
user_fraction = 0.01
the_one_percent = FractionalUserBase(contact_network,user_fraction)
print(len(the_one_percent.contact_network.nodes))


# create a user base from a Contiguous region about a random seed user (or a specified one)
user_fraction = 0.01
contiguous_one_percent = ContiguousUserBase(contact_network,user_fraction, seed_user=None)
print(len(contiguous_one_percent.contact_network.nodes))