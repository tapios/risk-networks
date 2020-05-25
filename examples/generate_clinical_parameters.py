# In this example, we generate clinical statistics for a sample
# population of size 100.

# Get the epiforecast package onto our path:
import os, sys; sys.path.append(os.path.join(".."))

# Import utilities for generating random populations
from epiforecast.populations import populate_ages, ClinicalStatistics, TransitionRates
from epiforecast.samplers import GammaSampler, AgeAwareBetaSampler

# First we define the population by the number of individuals
population = 100

# ... and the age category of each individual
age_distribution = [ 0.23,  # 0-19 years
                     0.39,  # 20-44 years
                     0.25,  # 45-64 years
                     0.079  # 65-75 years
                    ]

# 75 onwards
age_distribution.append(1 - sum(age_distribution))

ages = populate_ages(population, distribution=age_distribution)

# Print the ages we randomly generated:
print("Ages categories of a random population:\n", ages, "\n")

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

# Print the latent periods
print("Randomly-generated latent periods:\n", latent_periods.values)

# We process the clinical data to determine transition rates between each epidemiological state,

transition_rates = TransitionRates(latent_periods,
                                   community_infection_periods,
                                   hospital_infection_periods,
                                   hospitalization_fraction,
                                   community_mortality_fraction,
                                   hospital_mortality_fraction)

# The transition rates have units 1 / day. There are six transition rates. For example:
print("\nInfected -> Deceased transition rates (1/days):\n")
for node in transition_rates.nodes:
    print("node ", node, ": ", transition_rates.infected_to_deceased[node])
