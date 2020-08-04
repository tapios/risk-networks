import numpy as np
from epiforecast.measurements import (Observation,
                                      FixedObservation,
                                      BudgetedObservation,
                                      StaticNeighborObservation,
                                      DataObservation,
                                      HighVarianceObservation)

from _argparse_init import arguments
from _user_network_init import user_population, user_nodes
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
# imperfect observations #######################################################
sensor_wearers=np.random.choice(user_nodes, size=arguments.observations_I_budget, replace=False)
continuous_infection_test = FixedObservation(
    N=user_population,
    obs_nodes=sensor_wearers,
    obs_status='I',
    obs_name="continuous_infection_test",
    noisy_measurement=True,
    sensitivity=0.5,
    specificity=0.5,
    obs_var_min=1e-6)

random_infection_test = Observation(
        N=user_population,
        obs_frac=arguments.observations_I_fraction_tested,
        obs_status='I',
        obs_name="Random Infection Test",
        min_threshold=arguments.observations_I_min_threshold,
        max_threshold=arguments.observations_I_max_threshold,
        noisy_measurement=True,
        sensitivity=0.99,
        obs_var_min=1e-6)

budgeted_random_infection_test = BudgetedObservation(
        N=user_population,
        obs_budget=arguments.observations_I_budget,
        obs_status='I',
        obs_name="Budgeted Infection Test",
        min_threshold=arguments.observations_I_min_threshold,
        max_threshold=arguments.observations_I_max_threshold,
        noisy_measurement=True,
        sensitivity=0.99,
        obs_var_min=1e-6)

neighbor_transfer_infection_test = StaticNeighborObservation(
        N=user_population,
        obs_budget=arguments.observations_I_budget,
        obs_status='I',
        obs_name="Static Neighbor Transfer Infection Test",
        storage_type="temporary",
        nbhd_sampling_method="random",
        noisy_measurement=True,
        sensitivity=0.99,
        obs_var_min=1e-6)

high_var_infection_test = HighVarianceObservation(
        N=user_population,
        obs_frac=arguments.observations_I_fraction_tested,
        obs_status='I',
        obs_name="Test maximal variance infected",
        noisy_measurement=True,
        sensitivity=0.99,
        obs_var_min=1e-6)

# perfect observations #########################################################
positive_hospital_records = DataObservation(
        N=user_population,
        set_to_one=True,
        obs_status='H',
        obs_name="positive_hospital_records")

negative_hospital_records = DataObservation(
        N=user_population,
        set_to_one=False,
        obs_status='H',
        obs_name="negative_hospital_records")

positive_death_records = DataObservation(
        N=user_population,
        set_to_one=True,
        obs_status='D',
        obs_name="positive_death_records")

negative_death_records = DataObservation(
        N=user_population,
        set_to_one=False,
        obs_status='D',
        obs_name="negative_death_records")

################################################################################
print_end_of(__name__)

