import numpy as np
import networkx as nx

class AgeDistribution(dict):
    def __init__(self, 
                     children, #  0 -- 19
                 young_adults, # 20 -- 44
                  middle_aged, # 45 -- 64
                      seniors, # 65 -- 75
                      elderly = None,  # 75 ->
                     quantile = False,
                 ):

        self.classes = 5

        self.children     = children
        self.young_adults = young_adults
        self.middle_aged  = middle_aged
        self.seniors      = seniors

        self.quantile     = quantile

        if quantile: # assume distribution sums to 1
            self.elderly = 1 - children - young_adults - middle_aged - seniors
        else:
            self.elderly = elderly

        self[0] = self.children
        self[1] = self.young_adults
        self[2] = self.middle_aged
        self[3] = self.seniors
        self[4] = self.elderly

        self.values = np.array([self.children, self.young_adults, self.middle_aged,
                                self.seniors, self.elderly])


def assign_ages(contact_network, age_distribution):
    age_classes = age_distribution.classes

    ages = { node: np.random.choice(np.arange(age_classes), p=age_distribution.values)
             for node in contact_network.nodes() }

    nx.set_node_attributes(contact_network, values=ages, name='age')

def king_county_distributions():
    """
    Returns the `AgeDistribution`s `population_quantiles, hospitalization_rates, 
    community_recovery_rates, hospital_recovery_rates` corresponding to 
    plausible estimates for King County, Washington State, United States of America.
    """

    population_quantiles     = AgeDistribution(    children = 0.23, 
                                               young_adults = 0.39, 
                                                middle_aged = 0.25, 
                                                    seniors = 0.79,
                                                   quantile = True)

    hospitalization_rates    = AgeDistribution(    children = 0.02, 
                                               young_adults = 0.17, 
                                                middle_aged = 0.25, 
                                                    seniors = 0.35, 
                                                    elderly = 0.45)

    community_recovery_rates = AgeDistribution(    children = 1e-15, 
                                               young_adults = 1e-3, 
                                                middle_aged = 5e-3, 
                                                    seniors = 0.02, 
                                                    elderly = 0.05)

    hospital_recovery_rates  = AgeDistribution(    children = 1e-15, 
                                               young_adults = 1e-3, 
                                                middle_aged = 0.01, 
                                                    seniors = 0.04, 
                                                    elderly = 0.1)

    return (population_quantiles, hospitalization_rates, 
            community_recovery_rates, hospital_recovery_rates)

