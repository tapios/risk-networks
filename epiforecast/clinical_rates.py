import networkx as nx
import numpy as np

from .samplers import DisplacedGammaSampler

# The five clinical rates are
#
# 1. Exposed      -> Infected
# 2. Infected     -> Resistant
# 3. Hopsitalized -> Resistant
# 4. Infected     -> Deceased
# 5. Hopsitalized -> Deceased

class ConstantClinicalProperty(dict):
    def __init__(self, constant):
        self.property = constant
    def __missing__(self, key):
        return self.property

class VariableClinicalProperty(dict):
    def __init__(self, contact_network, generator):
        self.properties = { node: generator(node) for node in contact_network.nodes() }
        self.update(self.properties)

class TransitionRates:
    def __init__(self, contact_network,
                                      latent_period = DisplacedGammaSampler(2, 1.7, 2),
                    fractional_hospitalization_rate = DisplacedGammaSampler(1, 1.5, 2),
                         community_infection_period = DisplacedGammaSampler(1, 1.5, 2),
                          hospital_infection_period = DisplacedGammaSampler(1, 1.5, 3),
                 fractional_community_morality_rate = ,
                  fractional_hospital_morality_rate = ,
                 ):
