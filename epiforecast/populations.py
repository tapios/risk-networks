import numpy as np

from .samplers import AgeAwareBetaSampler, GammaSampler

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

    Args:

    latent_period of infection (1/σ)

    community_infection_period over which infection persists in the 'community' (1/γ),

    hospital_infection_period over which infection persists in a hospital setting (1/γ′),

    hospitalization_fraction, the fraction of infected that become hospitalized (h),

    community_mortality_fraction, the mortality rate in the community (d),

    hospital_mortality_fraction, the mortality rate in a hospital setting (d′).

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

        self.population = len(latent_periods.values)
        self.nodes = nodes = range(self.population)

        σ = 1 / latent_periods.values
        γ = 1 / community_infection_periods.values
        h = hospitalization_fraction.values
        d = community_mortality_fraction.values

        γ_prime = 1 / hospital_infection_periods.values
        d_prime = hospital_mortality_fraction.values

        self.exposed_to_infected       = { node: σ[node]                             for node in nodes }
        self.infected_to_resistant     = { node: (1 - h[node] - d[node]) * γ[node]   for node in nodes }
        self.infected_to_hospitalized  = { node: h[node] * γ[node]                   for node in nodes }
        self.infected_to_deceased      = { node: d[node] * γ[node]                   for node in nodes }

        self.hospitalized_to_resistant = { node: (1 - d_prime[node]) * γ_prime[node] for node in nodes }
        self.hospitalized_to_deceased  = { node: d_prime[node] * γ_prime[node]       for node in nodes }


def king_county_transition_rates(population):

    # ... and the age category of each individual
    age_distribution = [ 0.23,  # 0-19 years
                         0.39,  # 20-44 years
                         0.25,  # 45-64 years
                         0.079  # 65-75 years
                        ]

    # 75 onwards
    age_distribution.append(1 - sum(age_distribution))
    
    ages = populate_ages(population, distribution=age_distribution)
    
    # Next, we randomly generate clinical properties for our example population.
    # Note that the units of 'periods' are days, and the units of 'rates' are 1/day.
    latent_periods = ClinicalStatistics(ages = ages, minimum = 2,
                                                     sampler = GammaSampler(k=1.7, theta=2.0))
    
    community_infection_periods = ClinicalStatistics(ages = ages, minimum = 1,
                                                     sampler = GammaSampler(k=1.5, theta=2.0))
    
    hospital_infection_periods = ClinicalStatistics(ages = ages, minimum = 1,
                                                    sampler = GammaSampler(k=1.5, theta=3.0))
    
    hospitalization_fraction = ClinicalStatistics(ages = ages,
        sampler = AgeAwareBetaSampler(mean=[ 0.02,  0.17,  0.25, 0.35, 0.45], b=4))
    
    community_mortality_fraction = ClinicalStatistics(ages = ages,
        sampler = AgeAwareBetaSampler(mean=[0.001, 0.001, 0.005, 0.02, 0.05], b=4))
    
    hospital_mortality_fraction  = ClinicalStatistics(ages = ages,
        sampler = AgeAwareBetaSampler(mean=[0.001, 0.001,  0.01, 0.04,  0.1], b=4))

    transition_rates = TransitionRates(latent_periods,
                                       community_infection_periods,
                                       hospital_infection_periods,
                                       hospitalization_fraction,
                                       community_mortality_fraction,
                                       hospital_mortality_fraction)
    
    return transition_rates





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
