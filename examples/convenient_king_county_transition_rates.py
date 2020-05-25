# In this example, we generate clinical statistics for a sample
# population of size 100.

# Get the epiforecast package onto our path:
import os, sys; sys.path.append(os.path.join(".."))

# Import utilities for generating random populations
from epiforecast.populations import king_county_transition_rates

# First we define the population by the number of individuals
population = 100

transition_rates = king_county_transition_rates(population)

# The transition rates have units 1 / day. There are six transition rates. For example:
print("\nInfected -> Deceased transition rates (1/days):\n")
for node in transition_rates.nodes:
    print("node ", node, ": ", transition_rates.infected_to_deceased[node])

