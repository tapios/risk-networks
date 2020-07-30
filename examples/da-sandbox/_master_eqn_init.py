import numpy as np

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
n_backward_steps = 2 # minimum amount of steps per time step: backward run

transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_particle = transition_rates[user_nodes]
    transition_rates_particle.calculate_from_clinical()
    transition_rates_ensemble.append(transition_rates_particle)

community_transmission_rate_ensemble = np.full([ensemble_size, 1],
                                               community_transmission_rate)

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

