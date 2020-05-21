# Get the epiforecast package onto our path:
import os, sys; sys.path.append(os.path.join(".."))

from epiforecast.scenarios import random_infection
from epiforecast.scenarios import midnight_on_Tuesday
#from epiforecast.scenarios import state_distribution_at_midnight_on_Tuesday
#from epiforecast.scenarios import transition_rates_distribution_at_midnight_on_Tuesday
#from epiforecast.scenarios import transmission_rates_distribution_at_midnight_on_Tuesday

population = 100

initial_state = random_infection(population, infected=20)

print("\n \n An initial random infection (squint to see the 1's)")
print(initial_state)

class DummyKineticModel:
    """ A dummy class for a kinetic model until the real one arrives. """
    def __init__(self, population):
        self.population = population

kinetic_model = DummyKineticModel(population)

# Instantiate an example epidemic state
state = midnight_on_Tuesday(kinetic_model)

print("\n \n An epidemic, at midnight on Tuesday:")
print(state)
