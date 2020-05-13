import os, sys

sys.path.append(os.getenv('EPIFORECAST', os.path.join("..")))

import numpy as np
import networkx as nx
import epiforecast

# Load a sample contact network
contact_network = epiforecast.load_sample_contact_network()

# Instantiate a model on the sample network
model = epiforecast.StaticRiskNetworkModel(contact_network)

# Set to default transition rates
model.set_transition_rates(epiforecast.TransitionRates())

# Initialize a state with a small, randomly-selected number of infected.
epiforecast.random_infection(model)

# Integrate the model forwards to time = 1.0
residual = model.integrate_forwards(1.0)
