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

