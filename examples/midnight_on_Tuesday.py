# Get the epiforecast package onto our path:
import os, sys; sys.path.append(os.path.join(".."))


# Tools for simulating specific scenarios
from epiforecast.scenarios import random_infection, randomly_infected_ensemble
from epiforecast.scenarios import midnight_on_Tuesday
from epiforecast.scenarios import percent_infected_at_midnight_on_Tuesday
from epiforecast.scenarios import ensemble_transition_rates_at_midnight_on_Tuesday
from epiforecast.scenarios import ensemble_transmission_rates_at_midnight_on_Tuesday

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

ensemble_size = 3
population = 10

# Generate a joint distribution of states and transition rates for this example.
ensemble_states = randomly_infected_ensemble(ensemble_size, population, 
                                             percent_infected_at_midnight_on_Tuesday())

print("\n \n Initial ensemble states for data assimilation at midnight on Tuesday:")
print(ensemble_states)

ensemble_transition_rates = ensemble_transition_rates_at_midnight_on_Tuesday(ensemble_size, population)

print("\n \n Initial ensemble latent periods for data assimilation at midnight on Tuesday:")
[print(rates.latent_periods) for rates in ensemble_transition_rates]

ensemble_transmission_rates = ensemble_transmission_rates_at_midnight_on_Tuesday(ensemble_size)

print("\n \n Initial ensemble transmission rates at midnight on Tuesday:")
print(ensemble_transmission_rates)
