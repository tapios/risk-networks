import os, sys; sys.path.append(os.path.join(".."))
from epiforecast.epiplots import plot_master_eqns

from epiforecast.populations import populate_ages, sample_distribution, TransitionRates

from epiforecast.samplers import GammaSampler, BetaSampler
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler

from epiforecast.observations import FullObservation, HighProbRandomStatusObservation
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.measurements import TestMeasurements

import numpy as np
import networkx as nx
np.random.seed(1)

population = 1000
contact_network = nx.watts_strogatz_graph(population, 12, 0.1, 1)
ensemble_size = 10


latent_periods              = sample_distribution(GammaSampler(k=1.7, theta=2.0), population=population, minimum=2)
community_infection_periods = sample_distribution(GammaSampler(k=1.5, theta=2.0), population=population, minimum=1)
hospital_infection_periods  = sample_distribution(GammaSampler(k=1.5, theta=3.0), population=population, minimum=1)

hospitalization_fraction     = sample_distribution(BetaSampler(mean=0.25, b=4), population=population)
community_mortality_fraction = sample_distribution(BetaSampler(mean=0.02, b=4), population=population)
hospital_mortality_fraction  = sample_distribution(BetaSampler(mean=0.04, b=4), population=population)

transmission_rate = 0.06*np.ones(ensemble_size)
transition_rates = TransitionRates(contact_network,
                                   latent_periods,
                                   community_infection_periods,
                                   hospital_infection_periods,
                                   hospitalization_fraction,
                                   community_mortality_fraction,
                                   hospital_mortality_fraction)

ensemble_model = MasterEquationModelEnsemble(contact_network,
                [transition_rates]*ensemble_size,
                transmission_rate,
                ensemble_size = ensemble_size)

np.random.seed(1)

I_perc = 0.01
y0      = np.zeros([ensemble_size, 5 * population])

for mm, member in enumerate(ensemble_model.ensemble):
    infected = np.random.choice(population, replace = False, size = int(population * I_perc))
    E, I, H, R, D = np.zeros([5, population])
    S = np.ones(population,)
    I[infected] = 1.
    S[infected] = 0.

    y0[mm, : ]  = np.hstack((S, I, H, R, D))

tF = 35
res = ensemble_model.simulate(y0, tF, n_steps = 100)

ode_states = res['states'][:,:,-1]

def random_state(population):
    """
    Returns a status dictionary associated with a random infection
    """
    status_catalog = ['S', 'E', 'I', 'H', 'R', 'D']
    status_weights = [0.8, 0.01, 0.15, 0.02, 0.01, 0.01]
    statuses = {node: status_catalog[np.random.choice(6, p = status_weights)] for node in range(population)}

    return statuses

statuses = random_state(population)

test = TestMeasurements()
test.update_prevalence(ode_states, scale = None)
mean, var = test.take_measurements(statuses, scale = None)

# print(statuses.values())

print('\n1st Test: Probs in natural scale ----------------------------')
print(np.vstack([np.array(list(statuses.values())), mean, var]).T[:5])

test = TestMeasurements()
test.update_prevalence(ode_states)
mean, var = test.take_measurements(statuses)

print('\n2nd Test: Probs in logit scale ------------------------------')
print(np.vstack([np.array(list(statuses.values())), mean, var]).T[:5])

print('\n3rd Test: Probing for wrong status --------------------------')
mean, var = test.take_measurements(statuses, status = 'E')
