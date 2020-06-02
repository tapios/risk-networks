import os, sys; sys.path.append(os.path.join(".."))
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.populations import populate_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, BetaSampler
from epiforecast.epiplots import plot_master_eqns

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

latent_periods              = sample_distribution(GammaSampler(k=1.7, theta=2.0), population=population, minimum=2)
community_infection_periods = sample_distribution(GammaSampler(k=1.5, theta=2.0), population=population, minimum=1)
hospital_infection_periods  = sample_distribution(GammaSampler(k=1.5, theta=3.0), population=population, minimum=1)

hospitalization_fraction     = sample_distribution(BetaSampler(mean=0.25, b=4), population=population)
community_mortality_fraction = sample_distribution(BetaSampler(mean=0.02, b=4), population=population)
hospital_mortality_fraction  = sample_distribution(BetaSampler(mean=0.04, b=4), population=population)

transition_rates = TransitionRates(contact_network,
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

# ------------------------------------------------------------------------------
# Fifth test: create object with adjacency matrix
master_eqn_ensemble = MasterEquationModelEnsemble(nx.to_scipy_sparse_matrix(contact_network),
            [transition_rates]*ensemble_size,
            transmission_rate,
            ensemble_size = ensemble_size)
print('Fifth test: passed')

# ------------------------------------------------------------------------------
# Fifth test: simulate the epidemic through the master equations
np.random.seed(1)

I_perc = 0.01
y0 = np.zeros([ensemble_size, 5 * population])

for mm, member in enumerate(master_eqn_ensemble.ensemble):
    infected = np.random.choice(population, replace = False, size = int(population * I_perc))
    E, I, H, R, D = np.zeros([5, population])
    S = np.ones(population,)
    I[infected] = 1.
    S[infected] = 0.

    y0[mm, : ] = np.hstack((S, I, H, R, D))

res = master_eqn_ensemble.simulate(y0, 100, n_steps = 20)
print('Simulation done!')

# fig, axes = plot_master_eqns(res['states'], res['times'])

# current_state = np.random.uniform(0.01, 0.2, size=(ensemble_size,5*population))
# static_network_interval=0.25
# new_state = master_eqn_ensemble.simulate(current_state, static_network_interval)
#can optionally
#n_steps=number of steps(=100), t_0=initial_time(=0), and closure(='independent')


# in practice when we update we will always update rates and network at the same time:
# new_contact_network = nx.watts_strogatz_graph(population, 12, 0.1, 2)
# new_transmission_rate = 0.08*np.ones(ensemble_size)
# master_eqn_ensemble.update_ensemble(new_contact_network=new_contact_network,
#                                     new_transition_rates=new_transition_rates,
#                                     new_transmission_rate=new_transmission_rate)



#verification test?
