import numpy as np

from epiforecast.risk_simulator import MasterEquationModelEnsemble

from _argparse_init import arguments
from _constants import (start_time,
                        community_transmission_rate,
                        hospital_transmission_reduction)
from _stochastic_init import transition_rates
from _user_network_init import user_nodes, user_population, user_network
from _utilities import print_start_of, print_end_of

from epiforecast.populations import TransitionRates
from epiforecast.samplers import BetaSampler, GammaSampler


print_start_of(__name__)
################################################################################
ensemble_size = 100
n_forward_steps  = 1 # minimum amount of steps per time step: forward run
n_backward_steps = 8 # minimum amount of steps per time step: backward run

# Prior of transition rates ####################################################
learn_transition_rates = arguments.params_learn_transition_rates
transition_rates_ensemble = []
if learn_transition_rates == True:
    parameter_str = arguments.params_transition_rates_str.split(',')
    for i in range(ensemble_size):
        transition_rates = TransitionRates.from_samplers(
                population=user_network.get_node_count(),
                lp_sampler=GammaSampler(1.7,2.,1.),
                cip_sampler=GammaSampler(1.5,2.,1.),
                hip_sampler=GammaSampler(1.5,3.,1.),
                hf_sampler=BetaSampler(4.,0.036),
                cmf_sampler=BetaSampler(4.,0.001),
                hmf_sampler=BetaSampler(4.,0.18)
        )
        transition_rates.calculate_from_clinical()
        transition_rates_particle = transition_rates[user_nodes]
        transition_rates_particle.calculate_from_clinical()
        transition_rates_ensemble.append(transition_rates_particle)

else:
    parameter_str = None
    for i in range(ensemble_size):
        transition_rates_particle = transition_rates[user_nodes]
        transition_rates_particle.calculate_from_clinical()
        transition_rates_ensemble.append(transition_rates_particle)

# range of transition rates
transition_rates_min = {
        'latent_periods': 2,
        'community_infection_periods': 1,
        'hospital_infection_periods': 1,
        'hospitalization_fraction': 1e-5,
        'community_mortality_fraction': 0,
        'hospital_mortality_fraction': 0
}

transition_rates_max = {
        'latent_periods': 12,
        'community_infection_periods': 15,
        'hospital_infection_periods': 10,
        'hospitalization_fraction': 0.99999,
        'community_mortality_fraction': 1,
        'hospital_mortality_fraction': 1
}

# Prior of transmission rate ###################################################
community_transmission_rate_ensemble = np.full([ensemble_size, 1],
                                               community_transmission_rate)
learn_transmission_rate = arguments.params_learn_transmission_rate
transmission_rate_bias = arguments.params_transmission_rate_bias
transmission_rate_std = arguments.params_transmission_rate_noise * community_transmission_rate
if learn_transmission_rate == True:
    community_transmission_rate_ensemble = np.random.lognormal(
                           np.log(community_transmission_rate + transmission_rate_bias),
                           np.log(transmission_rate_std),
                           community_transmission_rate_ensemble.shape)

# range of transmission rate
transmission_rate_min = 3
transmission_rate_max = 21

# Set up master equation solver ################################################
master_eqn_ensemble = MasterEquationModelEnsemble(
        population=user_population,
        transition_rates=transition_rates_ensemble,
        transmission_rate=community_transmission_rate_ensemble,
        hospital_transmission_reduction=hospital_transmission_reduction,
        ensemble_size=ensemble_size,
        start_time=start_time,
        parallel_cpu=arguments.parallel_flag,
        num_cpus=arguments.parallel_num_cpus
)

ensemble_ic = np.zeros([ensemble_size, 5*user_population])
ensemble_ic[:,user_population:2*user_population] = np.random.beta(arguments.ic_alpha,
                                                                  arguments.ic_beta,
                                                                  (ensemble_size, user_population))
ensemble_ic[:,:user_population] = 1 - ensemble_ic[:,user_population:2*user_population]

################################################################################
print_end_of(__name__)

