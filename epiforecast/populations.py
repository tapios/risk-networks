import numpy as np

def sample_clinical_distribution(sampler, ages=None, population=None, minimum=0):
    """
    Generate clinical parameters by sampling from a distribution.

    Use cases
    --------

    1. `ages` is not `None`: assume `population = len(ages)`; return an array of size `len(ages)`
       of clinical parameter samples using `minimum + sampler.draw(age)`.

    2. `ages` is None, `population` is not `None`: return an array of size `population` of
       clinical parameter samples using `mininum + sampler.draw()`.

    3. Both `ages` and `population` are `None`: return a single `minimum + sampler.draw()`.

    Args
    ----

    ages (list-like): a list of age categories for the population

    minimum: the minimum value of the statistic (note that this assumes 
             `sampler.draw(age) is always greater than 0.)
             
    sampler: a 'sampler' with a function `sampler.draw(age)` that draws a random
             sample from a distribution, depending on `age`. Samplers that are
             age-independent must still support the syntax `sampler.draw(age)`. 
    """

    if ages is not None:
        return np.array([minimum + sampler.draw(age) for age in ages])
    elif population is not None:
        return np.array([minimum + sampler.draw() for i in range(population)])
    else:
        return minimum + sampler.draw()


class TransitionRates:
    """
    A container for transition rates.

    Args
    ----

    All arguments are arrays that reflect the clinical parameters of a population:

    * latent_period of infection (1/σ)

    * community_infection_period over which infection persists in the 'community' (1/γ),

    * hospital_infection_period over which infection persists in a hospital setting (1/γ′),

    * hospitalization_fraction, the fraction of infected that become hospitalized (h),

    * community_mortality_fraction, the mortality rate in the community (d),

    * hospital_mortality_fraction, the mortality rate in a hospital setting (d′).

    * population, the size of population (not always inferrable from clinical parameters)
    
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
    def __init__(self,
                 population,
                 latent_periods,
                 community_infection_periods,
                 hospital_infection_periods,
                 hospitalization_fraction,
                 community_mortality_fraction,
                 hospital_mortality_fraction):

        # For data assimilation we require return of initial variables
        self.latent_periods               = latent_periods
        self.community_infection_periods  = community_infection_periods
        self.hospital_infection_periods   = hospital_infection_periods
        self.hospitalization_fraction     = hospitalization_fraction
        self.community_mortality_fraction = community_mortality_fraction
        self.hospital_mortality_fraction  = hospital_mortality_fraction

        self.population = population 
        self.nodes = nodes = range(self.population)

        self._calculate_transition_rates()

    def _calculate_transition_rates(self):
        """
        Calculates the transition rates, given the current clinical parameters.
        If the clinical parameter is only a single value, we apply it to all nodes.
        """
        σ = 1 / self.latent_periods
        γ = 1 / self.community_infection_periods
        h = self.hospitalization_fraction
        d = self.community_mortality_fraction
        γ_prime = 1 / self.hospital_infection_periods
        d_prime = self.hospital_mortality_fraction

        # Broadcast to arrays of size `population`
        σ *= np.ones(self.population)
        γ *= np.ones(self.population)
        h *= np.ones(self.population)
        d *= np.ones(self.population)
        γ_prime *= np.ones(self.population)
        d_prime *= np.ones(self.population)

        self.exposed_to_infected       = { node: σ[node]                             for node in self.nodes }
        self.infected_to_resistant     = { node: (1 - h[node] - d[node]) * γ[node]   for node in self.nodes }
        self.infected_to_hospitalized  = { node: h[node] * γ[node]                   for node in self.nodes }
        self.infected_to_deceased      = { node: d[node] * γ[node]                   for node in self.nodes }

        self.hospitalized_to_resistant = { node: (1 - d_prime[node]) * γ_prime[node] for node in self.nodes }
        self.hospitalized_to_deceased  = { node: d_prime[node] * γ_prime[node]       for node in self.nodes }

    def set_clinical_parameter(self, parameter, values):
        setattr(self, parameter, values)
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
