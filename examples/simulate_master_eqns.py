import os, sys; sys.path.append(os.path.join(".."))
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.populations import populate_ages, ClinicalStatistics, TransitionRates
from epiforecast.samplers import GammaSampler, AgeAwareBetaSampler

import numpy as np
import networkx as nx

population = 1000
# define the contact network [networkx.graph] - haven't done this yet
contact_network = nx.watts_strogatz_graph(population, 12, 0.1, 1)

# give the ensemble size (is this required)
ensemble_size = 10

# ------------------------------------------------------------------------------
# First test: an empty ensemble with a given graph (no rates)
master_eqn_ensemble = MasterEquationModelEnsemble(contact_network, None, None, ensemble_size = 10)
print('First test: passed')

# ------------------------------------------------------------------------------
# Second test: create the transmission rate in the ensemble
transmission_rate = 0.06*np.ones(ensemble_size)
master_eqn_ensemble.update_transmission_rate(transmission_rate)
print('Second test: passed')

# ------------------------------------------------------------------------------
# Third test: create transition rates and populate the ensemble
# Simple setting: one age group all ensemble members are identical
age_distribution = [ 1. ]
ages = populate_ages(population, distribution=age_distribution)

latent_periods = ClinicalStatistics(ages = ages, minimum = 2,
                                    sampler = GammaSampler(k=1.7, theta=2.0))

community_infection_periods = ClinicalStatistics(ages = ages, minimum = 1,
                                    sampler = GammaSampler(k=1.5, theta=2.0))

hospital_infection_periods = ClinicalStatistics(ages = ages, minimum = 1,
                                    sampler = GammaSampler(k=1.5, theta=3.0))

hospitalization_fraction = ClinicalStatistics(ages = ages,
    sampler = AgeAwareBetaSampler(mean=[ 0.25 ], b=4))

community_mortality_fraction = ClinicalStatistics(ages = ages,
    sampler = AgeAwareBetaSampler(mean=[0.02], b=4))

hospital_mortality_fraction  = ClinicalStatistics(ages = ages,
    sampler = AgeAwareBetaSampler(mean=[0.04], b=4))

transition_rates = TransitionRates(population,
                                   latent_periods,
                                   community_infection_periods,
                                   hospital_infection_periods,
                                   hospitalization_fraction,
                                   community_mortality_fraction,
                                   hospital_mortality_fraction)

master_eqn_ensemble.update_transition_rates([transition_rates]*master_eqn_ensemble.M)
print('Third test: passed')
# ------------------------------------------------------------------------------
# Fourth test: create object with all parameters at once
master_eqn_ensemble = MasterEquationModelEnsemble(contact_network,
            [transition_rates]*ensemble_size,
            transmission_rate,
            ensemble_size = ensemble_size)
print('Fourth test: passed')



# simulate
current_state = np.random.uniform(0.01, 0.2, size=(ensemble_size,5*population))
static_network_interval=0.25
new_state = master_eqn_ensemble.simulate(current_state, static_network_interval)
#can optionally
#n_steps=number of steps(=100), t_0=initial_time(=0), and closure(='independent')


# in practice when we update we will always update rates and network at the same time:
new_contact_network = nx.watts_strogatz_graph(population, 12, 0.1, 2)
new_transmission_rate = 0.08*np.ones(ensemble_size)
master_eqn_ensemble.update_ensemble(new_contact_network=new_contact_network,
                                    new_transition_rates=new_transition_rates,
                                    new_transmission_rate=new_transmission_rate)



#verification test?
