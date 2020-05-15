import networkx as nx
import numpy as np

def beta_distributed_coin(scale=1, beta=4, p=0.05):
    return lambda: scale * np.random.beta(b * p / (1 - p), b=b)

class AgeDistribution(dict):
    def __init__(self, 
                     children, #  0 -- 19
                 young_adults, # 20 -- 44
                  middle_aged, # 45 -- 64
                      seniors, # 65 -- 75
                      elderly = None,  # 75 ->
                     quantile = False,
                 ):

        self.children     = children
        self.young_adults = young_adults
        self.middle_aged  = middle_aged
        self.seniors      = seniors
        self.quantile     = quantile

        if quantile: # assume distribution sums to 1
            self.elderly = 1 - children - young_adults - middle_aged - seniors
        else:
            self.elderly = elderly

        self['children']     = self.children
        self['young_adults'] = self.young_adults
        self['middle_aged']  = self.middle_aged
        self['seniors']      = self.seniors
        self['elderly']      = self.elderly


class InfectiousPopulation:
    def __init__(self, contact_network,
                       infection_rates = None, # aka, "neighbor-induced" transitions
                        clinical_rates = None, # aka, "spontaneous" transitions
                ):

        self.contact_network = contact_network
        self.clinical_rates = clinical_rates
        self.infection_rates = clinical_rates


# TODO: What is the correct name for this class?
class ScaledBetaDistribution:
    """
    A class representing a parameterized beta distribution.

    This class is used to model the distribution of infection
    rates among a population.

    The parameterized distribution is `Beta(a, b)`, where `a = b * p / (1 - p)`.

    The function `distribution.coin_flip()` takes a random draw
    from the parameterized distribution, and scales the result by `scale`.

    Args
    ----
    
    scale : Number by which to scale `coin_flip`.
        p : TODO: What do we call "p"?
        b : The 'beta' parameter in the Beta distribution.

    """
    def __init__(self, scale, p, b):
        self.scale = scale
        self.b = b # "beta" on Wikipedia
        self.p = p # TODO: correctly name this parameter.

    def coin_flip(self):
        return scale * np.random.beta(b * p / (1 - p), b=b)



class InfectionRateDistribution:
    def __init__(self, contact_network,
                    community_baseline = 0.05,
                     hospital_baseline = 0.1,
                 community_variability = ScaledBetaDistribution(scale=0.05, p=0.05, b=4)
                  hospital_variability = ScaledBetaDistribution(scale=0.75, p=0.05, b=4)
                 ):

        self.contact_network = contact_network
        self.initialize_infection_graph()

        self.generate_community_infection_rates(   baseline = community_baseline,
                                                variability = community_variability)

        self.generate_hospital_infection_rates(    baseline = hospital_baseline,
                                                variability = hospital_variability)

    def initialize_infection_graph(self):
        """
        Initialize the graph that represents 'induced' transitions
        from suceptible community members to infected community members
        due to contact with (1) an infected community member, or (2) a
        hospitalized person. 

        'Health workers' are community members that contact hopsitalized people.
        """
            
        # The infection graph is a small directed graph whose edges correspond
        # to the set of possible induced transitions.
        infection_graph = nx.DiGraph()

        # One edge representing transition from suspectible to exposed, 
        # 'induced' by an infected community member.

        # The rate is set to 1 because rate variability across the population
        # is set to the contact network.
        infection_graph.add_edge(('I', 'S'), ('I', 'E'), rate=1, weight_label='')

        # One edge representing a transition from suspectible to exposed,
        # 'induced' by a hopsitalized person.
        infection_graph.add_edge(('H', 'S'), ('H', 'E'), rate=1, weight_label='')

        self.infection_graph = infection_graph

    def generate_community_infection_rates(self, baseline=None, variability=None):
        """
        Generate a distribution of community infection rates with a constant
        `baseline` and random `variability` generated by `variability.coin_flip()`.
        """

        if baseline is None: 
            baseline = self.community_baseline
        else:
            self.community_baseline = baseline

        if variability is None: 
            variability = self.community_variability
        else:
            self.community_variability = variability

        self.community_distirubtion = { edge: baseline + variability.coin_flip()
                                        for edge in self.contact_network.edges() }

    def generate_hospital_infection_rates(self, baseline=None, variability=None):
        """
        Generate a distribution of hospital infection rates with a constant
        `baseline` and random `variability` generated by `variability.coin_flip()`.
        """

        if baseline is None: 
            baseline = self.hospital_baseline
        else:
            self.hospital_baseline = baseline

        if variability is None: 
            variability = self.hospital_variability
        else:
            self.hospital_variability = variability

        self.hospital_distirubtion = { edge: baseline + variability.coin_flip()
                                       for edge in self.contact_network.edges() }



def king_county_demographics():

    population_quantiles    = AgeDistribution(    children = 0.23, 
                                              young_adults = 0.39, 
                                               middle_aged = 0.25, 
                                                   seniors = 0.79,
                                                  quantile = True)

    hospitalization_rates   = AgeDistribution(    children = 0.02, 
                                              young_adults = 0.17, 
                                               middle_aged = 0.25, 
                                                   seniors = 0.35, 
                                                   elderly = 0.45)

    recovery_rates          = AgeDistribution(    children = 1e-15, 
                                              young_adults = 1e-3, 
                                               middle_aged = 5e-3, 
                                                   seniors = 0.02, 
                                                   elderly = 0.05)

    hospital_recovery_rates = AgeDistribution(    children = 1e-15, 
                                              young_adults = 1e-3, 
                                               middle_aged = 0.01, 
                                                   seniors = 0.04, 
                                                   elderly = 0.1)

    return population_quantiles, hospitalization_rates, recovery_rates, hospital_recovery_rates
