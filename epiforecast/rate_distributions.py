import numpy as np

def beta_distributed_coin(scale=1, beta=4, p=0.05):
    return lambda: scale * np.random.beta(b * p / (1 - p), b=b)

class AgeDistribution(dict):
    def __init__(self, 
                     children = 0.23,
                 young_adults = 0.39,
                  middle_aged = 0.25,
                      seniors = 0.079,
                      elderly = None,
                 ):

        self.children     = children
        self.young_adults = young_adults
        self.middle_aged  = middle_aged
        self.seniors      = seniors

        if elderly is None: # assume distribution sums to 1
            self.elderly      = 1 - children - young_adults - middle_aged - seniors
        else:
            self.elderly = elderly

        self['children']     = self.children
        self['young_adults'] = self.young_adults
        self['middle_aged']  = self.middle_aged
        self['seniors']      = self.seniors
        self['elderly']      = self.elderly


def king_county_demographics():

    population_quantiles = AgeDistribution(children=0.23, young_adults=0.39, middle_aged=0.25, seniors=0.79)

    hospitalization_rate = AgeDistribution(
        children=0.02, young_adults=0.17, middle_aged=0.25, seniors=0.35, elderly=0.45)

    recovery_rates = AgeDistribution(
        children=1e-15, young_adults=1e-3, middle_aged=5e-3, seniors=0.02, elderly=0.05)

    hospital_recovery_rates = AgeDistribution(
        children=1e-15, young_adults=1e-3, middle_aged=0.01, seniors=0.04, elderly=0.1)

    return population_quantiles, hospitalization_rate, recovery_rates, hospital_recovery_rates
