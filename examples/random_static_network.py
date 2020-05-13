import os, sys; sys.path.append(os.getenv('EPIFORECAST', os.path.join("..")))

import numpy as np

import networkx as nx

import epiforecast

# Instantiate a model with 1000 nodes
model = epiforecast.StaticRiskNetworkModel(N_nodes=1000)

# Set to default transition rates
model.set_transition_rates(epiforecast.TransitionRates())

contact_network = epiforecast.load_sample_contact_network()

model.set_contact_network(contact_network)
