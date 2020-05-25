import numpy as np

class ClinicalStatistics:
    """
    A container for clinical statistics.

    Args
    ----

    ages (list-like): a list of age categories for the population

    minimum: the minimum value of the statistic (note that this assumes 
             `sampler.draw(age) is always greater than 0.)
             
    sampler: a 'sampler' with a function `sampler.draw(age)` that draws a random
             sample from a distribution, depending on `age`. Samplers that are
             age-independent must still support the syntax `sampler.draw(age)`. 

    The six clinical statistics are

    1. latent_period of infection (1/σ)
    2. community_infection_period over which infection persists in the 'community' (1/γ),
    3. hospital_infection_period over which infection persists in a hospital setting (1/γ′),
    4. hospitalization_fraction, the fraction of infected that become hospitalized (h),
    5. community_mortality_fraction, the mortality rate in the community (d),
    6. hospital_mortality_fraction, the mortality rate in a hospital setting (d′).
    """
    def __init__(self, ages, minimum=0, sampler=None):

        if sampler is not None:
            self.values = np.array([minimum + sampler.draw(age) for age in ages])
        else:
            self.values = np.array([minimum for age in ages])

        self.population = len(ages)
        self.ages = ages
        self.minimum = minimum





class TransitionRates:
    """
    A container for transition rates.

    Args
    ----

    All arguments are ClinicalStatistics.

    * latent_period of infection (1/σ)

    * community_infection_period over which infection persists in the 'community' (1/γ),

    * hospital_infection_period over which infection persists in a hospital setting (1/γ′),

    * hospitalization_fraction, the fraction of infected that become hospitalized (h),

    * community_mortality_fraction, the mortality rate in the community (d),

    * hospital_mortality_fraction, the mortality rate in a hospital setting (d′).

    The six transition rates are

    1. Exposed -> Infected
    2. Infected -> Hospitalized
    3. Infected -> Resistant
    4. Hospitalized -> Resistant
    5. Infected -> Deceased
    6. Hospitalized -> Deceased

    These correspond to the dictionaries:

    1. transition_rates.exposed_to_infected
    2. transition_rates.infected_to_hospitalized
    3. transition_rates.infected_to_resistant
    4. transition_rates.hospitalized_to_resistant
    5. transition_rates.infected_to_deceased
    6. transition_rates.hospitalized_to_deceased
    """
    def __init__(self, latent_periods,
                       community_infection_periods,
                       hospital_infection_periods,
                       hospitalization_fraction,
                       community_mortality_fraction,
                       hospital_mortality_fraction):

        # For data assimilation we require return of initial variables
        self.latent_periods               = latent_periods.values
        self.community_infection_periods  = community_infection_periods.values
        self.hospital_infection_periods   = hospital_infection_periods.values
        self.hospitalization_fraction     = hospitalization_fraction.values
        self.community_mortality_fraction = community_mortality_fraction.values
        self.hospital_mortality_fraction  = hospital_mortality_fraction.values

        self.population = len(latent_periods.values)
        self.nodes = nodes = range(self.population)

        self._calculate_transition_rates()

    def _calculate_transition_rates(self):

        σ = 1 / self.latent_periods
        γ = 1 / self.community_infection_periods
        h = self.hospitalization_fraction
        d = self.community_mortality_fraction

        γ_prime = 1 / self.hospital_infection_periods
        d_prime = self.hospital_mortality_fraction

        self.exposed_to_infected       = { node: σ[node]                             for node in self.nodes }
        self.infected_to_resistant     = { node: (1 - h[node] - d[node]) * γ[node]   for node in self.nodes }
        self.infected_to_hospitalized  = { node: h[node] * γ[node]                   for node in self.nodes }
        self.infected_to_deceased      = { node: d[node] * γ[node]                   for node in self.nodes }

        self.hospitalized_to_resistant = { node: (1 - d_prime[node]) * γ_prime[node] for node in self.nodes }
        self.hospitalized_to_deceased  = { node: d_prime[node] * γ_prime[node]       for node in self.nodes }

    def set_clinical_statistic(self, statistic, values):
        setattr(self, statistic, values)
        self._calculate_transition_rates()











def populate_ages(population=1000, distribution=[0.25, 0.5, 0.25]):
    """
    Returns a numpy array of length `population`, with age categories from
    0 to len(`distribution`), where the elements of `distribution` specify
    the probability that an individual's age falls into the corresponding category.
    """
    classes = len(distribution)

    ages = [np.random.choice(np.arange(classes), p=distribution) 
            for person in range(population)]

    return np.array(ages)
