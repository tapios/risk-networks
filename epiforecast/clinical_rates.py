import networkx as nx
import numpy as np

# The five clinical rates are
#
# 1. Exposed      -> Infected
# 2. Infected     -> Resistant
# 3. Hopsitalized -> Resistant
# 4. Infected     -> Deceased
# 5. Hopsitalized -> Deceased

class ConstantClinicalRates:
    def __init__(self, constant_rates):
        self.rates = constant_rates

class VariableClinicalRate:
    def __init__(self, contact_network, rate_generator):
        self.rates = { node: rate_generator(node) for node in contact_network.nodes() }
