import os, sys; sys.path.append(os.path.join(".."))

import networkx as nx
import numpy as np
from matplotlib import pylab as pl

from epiforecast.user_base import FullUserBase, FractionalUserBase, ContiguousUserBase, contiguous_indicators
from epiforecast.scenarios import load_edges

np.random.seed(123)

#plot graphs? NB plotting is very slow for >1000 nodes
plotfigs=True


# ---- Create network
#1) from nx function:

#contact_network = nx.watts_strogatz_graph(100000, 12, 0.1, 1)
#population = len(contact_network)

#2) Or create from file:
edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e4_nobeds.txt')) 

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)

# ----


# create a full user base
full_user_base=FullUserBase(contact_network)
print("User base: Full")
print("number of nodes:", len(full_user_base.contact_network.nodes))
print("number of edges:", len(full_user_base.contact_network.edges))

# create a user base from a random fraction of the population
user_fraction = 0.2
fractional_user_base = FractionalUserBase(contact_network,user_fraction)
print("")
print("User base: ", user_fraction, " fraction of nodes, randomly chosen")
print("number of nodes:", len(fractional_user_base.contact_network.nodes))
print("number of edges:", len(fractional_user_base.contact_network.edges))

interior,boundary,mean_exterior_neighbors = contiguous_indicators(contact_network,fractional_user_base.contact_network)
print("number of interior nodes:", interior)
print("number of boundary nodes:", boundary)
print("average exterior neighbours of boundary node:", mean_exterior_neighbors)

# create a user base from a Contiguous region about a random seed user (or a specified one)
neighbor_user_base = ContiguousUserBase(contact_network,user_fraction, method="neighbor", seed_user=None)
print("")
print("User base:", user_fraction, " fraction of nodes, chosen using neighbor method")
print("number of nodes:", len(neighbor_user_base.contact_network.nodes))
print("number of edges:", len(neighbor_user_base.contact_network.edges))

interior,boundary,mean_exterior_neighbors = contiguous_indicators(contact_network,neighbor_user_base.contact_network)
print("number of interior nodes:", interior)
print("number of boundary nodes:", boundary)
print("average exterior neighbours of boundary node:", mean_exterior_neighbors)


# create a user base from a Contiguous region about a random seed user (or a specified one)
clique_user_base = ContiguousUserBase(contact_network,user_fraction, method="clique", seed_user=None)
print("")
print("User base:", user_fraction, " fraction of nodes, chosen using clique method")
print("number of nodes:", len(clique_user_base.contact_network.nodes))
print("number of edges:", len(clique_user_base.contact_network.edges))

interior,boundary,mean_exterior_neighbors = contiguous_indicators(contact_network,clique_user_base.contact_network)
print("number of interior nodes:", interior)
print("number of boundary nodes:", boundary)
print("average exterior neighbours of boundary node:", mean_exterior_neighbors)

#plot graph
if plotfigs:
    pl.figure(1,figsize=(10, 10), dpi=100)
    nx.draw_networkx(contact_network,                     node_color='k', with_labels=False, node_size=10, alpha=0.1)
    nx.draw_networkx(neighbor_user_base.contact_network,  node_color='r', with_labels=False, node_size=10)
    pl.title('neighborhood based contact network', fontsize=20)
    pl.savefig('neighbor_network.png')

    pl.figure(2,figsize=(10, 10), dpi=100)
    nx.draw_networkx(contact_network,                     node_color='k', with_labels=False, node_size=10, alpha=0.1)
    nx.draw_networkx(clique_user_base.contact_network,    node_color='r', with_labels=False, node_size=10)
    pl.title('clique based contact network',fontsize=20)
    pl.savefig('clique_network.png')

    
    pl.figure(3,figsize=(10, 10), dpi=100)
    nx.draw_networkx(contact_network,                      node_color='k', with_labels=False, node_size=10, alpha=0.1)
    nx.draw_networkx(fractional_user_base.contact_network, node_color='r', with_labels=False, node_size=10)
    pl.title('random subset contact network',fontsize=20)
    pl.savefig('random_network.png')

