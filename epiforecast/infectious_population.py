import networkx as nx
import numpy as np

from .populations import AgeDistribution
from .infection_rate_distribution import InfectionRateDistribution

#def rewire_network(contact_network, rewiring):
#
#    nx.set_edge_attributes(self.contact_network,
#                           values = rewiring.community,
#                             name = 'community infection rate')
#
#    nx.set_edge_attributes(self.contact_network,
#                           values = rewiring.hospital,
#                             name = 'hospital infection rate')

class InfectiousPopulation:
    def __init__(self, contact_network,
                       infection_rates, # aka, "neighbor-induced" transitions
                        clinical_rates, # aka, "spontaneous" transitions
                ):

        self.contact_network = contact_network
        self.clinical_rates = clinical_rates

        # Sets self.infection_rates
        self.set_infection_rates(infection_rates)

        # Sets self.infection_transition_graph
        self.initialize_infection_transition_graph()

    def set_infection_rates(self, infection_rates):
        """Set `self.infection_rates` and normalize the contact network."""

        self.infection_rates = infection_rates

        nx.set_edge_attributes(self.contact_network,
                               values = infection_rates.community,
                                 name = 'community infection rate')

        nx.set_edge_attributes(self.contact_network,
                               values = infection_rates.hospital,
                                 name = 'hospital infection rate')

    def initialize_infection_transition_graph(self):
        """
        Initialize the graph that represents transitions from 
        'suceptible' community members to 'exposed' community members
        due to contact with 

            1. An infected community member
            2. A hospitalized person (who are all infected)

        Notes 
        -----
            * These transitions are classified as 'induced' by Epidemics On Networks (`EoN`).
            * 'Health workers' are community members that contact hopsitalized people.

        """

        # The infection graph is a small directed graph whose edges correspond
        # to the set of possible induced transitions.
        infection_transition_graph = nx.DiGraph()

        # One edge representing transition from suspectible to exposed,
        # 'induced' by an infected community member.

        # The rate is set to 1 because rate variability across the population
        # is set to the contact network.
        infection_transition_graph.add_edge(('I', 'S'), ('I', 'E'), 
                                            rate=1, weight_label='Susceptible-to-Exposed, Infected-induced')

        # One edge representing a transition from suspectible to exposed,
        # 'induced' by a hopsitalized person.
        infection_transition_graph.add_edge(('H', 'S'), ('H', 'E'),
                                            rate=1, weight_label='Susceptible-to-Exposed, Hospitalized-induced')

        self.infection_transition_graph = infection_transition_graph
