import numpy as np
import pdb

from _argparse_init import arguments

if arguments.parallel_flag:
    #For parallel master equations
    from epiforecast.risk_simulator_parallel import MasterEquationModelEnsemble

else:
    #For serial master equations
    from epiforecast.risk_simulator import MasterEquationModelEnsemble

from _constants import (start_time,
                        community_transmission_rate,
                        hospital_transmission_reduction)
from _stochastic_init import transition_rates
from _user_network_init import user_nodes, user_population
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
ensemble_size = 100
n_forward_steps  = 1 # minimum amount of steps per time step: forward run
n_backward_steps = 5 # minimum amount of steps per time step: backward run

learn_transition_rates = arguments.learn_transition_rates
parameter_str = arguments.transition_rates_str.split(',') 
noise_level = [float(value) for value in arguments.transition_rates_noise.split(',')]
transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_particle = transition_rates[user_nodes]
    if learn_transition_rates == True:
        transition_rates_particle.add_noise_to_clinical_parameters(
                parameter_str, noise_level)
    transition_rates_particle.calculate_from_clinical()
    transition_rates_ensemble.append(transition_rates_particle)

community_transmission_rate_ensemble = np.full([ensemble_size, 1],
                                               community_transmission_rate)

learn_transmission_rate = arguments.learn_transmission_rate
transmission_rate_bias = arguments.transmission_rate_bias
transmission_rate_std = arguments.transmission_rate_noise * community_transmission_rate
if learn_transmission_rate == True:
    community_transmission_rate_ensemble = np.random.lognormal(
                           np.log(community_transmission_rate + transmission_rate_bias),
                           np.log(transmission_rate_std),
                           community_transmission_rate_ensemble.shape)

if arguments.parallel_flag:
    master_eqn_ensemble = MasterEquationModelEnsemble(
            population=user_population,
            transition_rates=transition_rates_ensemble,
            transmission_rate=community_transmission_rate_ensemble,
            hospital_transmission_reduction=hospital_transmission_reduction,
            ensemble_size=ensemble_size,
            start_time=start_time,
            ncores=arguments.num_cores
    )
else:
    master_eqn_ensemble = MasterEquationModelEnsemble(
            population=user_population,
            transition_rates=transition_rates_ensemble,
            transmission_rate=community_transmission_rate_ensemble,
            hospital_transmission_reduction=hospital_transmission_reduction,
            ensemble_size=ensemble_size,
            start_time=start_time
    )

################################################################################
print_end_of(__name__)

