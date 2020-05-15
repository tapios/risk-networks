from .samplers import ScaledBetaSampler

class ConstantInfectionRates:
    def __init__(self, contact_network, community_infection_rate, hospital_infection_rate):
        self.contact_network = contact_network
        self.community = community_infection_rate
        self.hospital = hospital_infection_rate

class VariableInfectionRates:
    def __init__(self, contact_network,
                    community_baseline = 0.05,
                     hospital_baseline = 0.1,
                 community_variability = ScaledBetaSampler(scale=0.05, p=0.05, b=4)
                  hospital_variability = ScaledBetaSampler(scale=0.75, p=0.05, b=4)
                 ):

        self.contact_network = contact_network
        self.initialize_infection_graph()

        # Sets self.community
        self.generate_community_infection_rates(   baseline = community_baseline,
                                                variability = community_variability)

        # Sets self.hospital
        self.generate_hospital_infection_rates(    baseline = hospital_baseline,
                                                variability = hospital_variability)

    def generate_community_infection_rates(self, baseline=None, variability=None):
        """
        Generate a distribution of community infection rates with a constant
        `baseline` and random `variability` generated by `variability.draw()`.
        """

        if baseline is None: 
            baseline = self.community_baseline
        else:
            self.community_baseline = baseline

        if variability is None: 
            variability = self.community_variability
        else:
            self.community_variability = variability

        self.community = { edge: baseline + variability.draw()
                           for edge in self.contact_network.edges() }

    def generate_hospital_infection_rates(self, baseline=None, variability=None):
        """
        Generate a distribution of hospital infection rates with a constant
        `baseline` and random `variability` generated by `variability.draw()`.
        """

        if baseline is None: 
            baseline = self.hospital_baseline
        else:
            self.hospital_baseline = baseline

        if variability is None: 
            variability = self.hospital_variability
        else:
            self.hospital_variability = variability

        self.hospital = { edge: baseline + variability.draw()
                          for edge in self.contact_network.edges() }
