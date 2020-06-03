import os, sys; sys.path.append(os.path.join(".."))
from epiforecast.epiplots import plot_master_eqns

from epiforecast.populations import populate_ages, sample_pathological_distribution, TransitionRates

from epiforecast.samplers import GammaSampler, BetaSampler
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler

from epiforecast.observations import FullObservation, HighProbRandomStatusObservation
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.risk_simulator import MasterEquationModelEnsemble

import numpy as np
import networkx as nx
np.random.seed(1)

population = 1000
contact_network = nx.watts_strogatz_graph(population, 12, 0.1, 1)
ensemble_size = 10


latent_periods              = sample_pathological_distribution(GammaSampler(k=1.7, theta=2.0), population=population, minimum=2)
community_infection_periods = sample_pathological_distribution(GammaSampler(k=1.5, theta=2.0), population=population, minimum=1)
hospital_infection_periods  = sample_pathological_distribution(GammaSampler(k=1.5, theta=3.0), population=population, minimum=1)

hospitalization_fraction     = sample_pathological_distribution(BetaSampler(mean=0.25, b=4), population=population)
community_mortality_fraction = sample_pathological_distribution(BetaSampler(mean=0.02, b=4), population=population)
hospital_mortality_fraction  = sample_pathological_distribution(BetaSampler(mean=0.04, b=4), population=population)

transmission_rate = 0.06*np.ones(ens.M)
transition_rates = TransitionRates(contact_network,
                                   latent_periods,
                                   community_infection_periods,
                                   hospital_infection_periods,
                                   hospitalization_fraction,
                                   community_mortality_fraction,
                                   hospital_mortality_fraction)

ensemble = MasterEquationModelEnsemble(contact_network,
                [transition_rates]*ensemble_size,
                transmission_rate,
                ensemble_size = ensemble_size)
