import os
import numpy as np
import networkx as nx

np.random.seed(1234)

def load_sample_contact_network(data_path = None):

    # The default sample contact network is taken from...
    if data_path is None:
        sample_edge_list = 'edge_list_SBM_1e3.txt'

        relative_data_path = os.path.join(os.getenv('EPIFORECAST', '..'),
                                          os.path.join('data', 'networks', sample_edge_list))

        data_path = os.path.abspath(relative_data_path)

    # Build the contact_network
    edges = np.loadtxt(data_path)
    contact_network = nx.Graph([tuple(e) for e in edges])

    # Relabel the network's nodes
    # TODO: Explain why the nodes need to be relabeled.
    nodes = list(contact_network.nodes())
    zipped_nodes = zip(nodes, range(len(nodes)))
    contact_network = nx.relabel_nodes(contact_network, dict(zipped_nodes))

    return contact_network
