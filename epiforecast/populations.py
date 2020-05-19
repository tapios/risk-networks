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

    1. latent_period of infection (σ⁻¹)
    2. community_infection_period over which infection persists in the 'community' (γ),
    3. hospital_infection_period over which infection persists in a hospital setting (γ′),
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
