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
sensor_wearers=np.random.choice(user_nodes, size=arguments.observations_sensor_wearers, replace=False)


# imperfect observations #######################################################
# sensor type observation
sensor_readings = FixedObservation(
    N=user_population,
    obs_nodes=sensor_wearers,
    obs_status='I',
    obs_name="continuous_infection_test",
    noisy_measurement=True,
    sensitivity=0.5,
    specificity=0.75,
    obs_var_min=1e-6)

# virus test type observations
# Molecular Diagnostic Test
MDT_result_delay = 1.0 # delay to results of the virus test
MDT_neighbor_test = StaticNeighborObservation(
        N=user_population,
        obs_budget=arguments.observations_I_budget,
        obs_status='I',
        obs_name="Static Neighbor Transfer Infection Test",
        storage_type="temporary",
        nbhd_sampling_method="random",
        noisy_measurement=True,
        sensitivity=0.95,
        specificity=0.99,
        obs_var_min=1e-6)
MDT_budget_random_test = BudgetedObservation(
        N=user_population,
        obs_budget=arguments.observations_I_budget,
        obs_status='I',
        obs_name="Budgeted Infection Test",
        min_threshold=arguments.observations_I_min_threshold,
        max_threshold=arguments.observations_I_max_threshold,
        noisy_measurement=True,
        sensitivity=0.95,
        specificity=0.99,
        obs_var_min=1e-6)

MDT_high_var_test = HighVarianceObservation(
        N=user_population,
        obs_frac=arguments.observations_I_fraction_tested,
        obs_status='I',
        obs_name="Test maximal variance infected MDT",
        noisy_measurement=True,
        sensitivity=0.95,
        specificity=0.99,
        obs_var_min=1e-6)

# Rapid Diagnostic Test
RDT_result_delay = 0.0 # delay to results of the virus test
RDT_budget_random_test = BudgetedObservation(
        N=user_population,
        obs_budget=arguments.observations_I_budget,
        obs_status='I',
        obs_name="85% sensitive Budgeted RDT",
        min_threshold=arguments.observations_I_min_threshold,
        max_threshold=arguments.observations_I_max_threshold,
        noisy_measurement=True,
        sensitivity=0.85,
        specificity=0.99,
        obs_var_min=1e-6)

poor_RDT_budget_random_test = BudgetedObservation(
        N=user_population,
        obs_budget=arguments.observations_I_budget,
        obs_status='I',
        obs_name="60% sensitive Budgeted RDT",
        min_threshold=arguments.observations_I_min_threshold,
        max_threshold=arguments.observations_I_max_threshold,
        noisy_measurement=True,
        sensitivity=0.6,
        specificity=0.99,
        obs_var_min=1e-6)

RDT_high_var_test = HighVarianceObservation(
        N=user_population,
        obs_frac=arguments.observations_I_fraction_tested,
        obs_status='I',
        obs_name="Test maximal variance infected RDT",
        noisy_measurement=True,
        sensitivity=0.85,
        specificity=0.99,
        obs_var_min=1e-6)


# generic test templates
continuous_infection_test = FixedObservation(
    N=user_population,
    obs_nodes=sensor_wearers,
    obs_status='I',
    obs_name="continuous_infection_test",
    noisy_measurement=True,
    sensitivity=0.5,
    specificity=0.75,
    obs_var_min=1e-6)

random_infection_test = Observation(
        N=user_population,
        obs_frac=arguments.observations_I_fraction_tested,
        obs_status='I',
        obs_name="Random Infection Test",
        min_threshold=arguments.observations_I_min_threshold,
        max_threshold=arguments.observations_I_max_threshold,
        noisy_measurement=True,
        sensitivity=0.95,
        specificity=0.99,
        obs_var_min=1e-6)

budgeted_random_infection_test = BudgetedObservation(
        N=user_population,
        obs_budget=arguments.observations_I_budget,
        obs_status='I',
        obs_name="Budgeted Infection Test",
        min_threshold=arguments.observations_I_min_threshold,
        max_threshold=arguments.observations_I_max_threshold,
        noisy_measurement=True,
        sensitivity=0.95,
        specificity=0.99,
        obs_var_min=1e-6)

neighbor_transfer_infection_test = StaticNeighborObservation(
        N=user_population,
        obs_budget=arguments.observations_I_budget,
        obs_status='I',
        obs_name="Static Neighbor Transfer Infection Test",
        storage_type="temporary",
        nbhd_sampling_method="random",
        noisy_measurement=True,
        sensitivity=0.95,
        specificity=0.99,
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
def transition_function(x, y, transition_steps, total_steps):
    """
    linear transition, stays at final value
    f(0) = x, 
    f(transition_steps) = y,
    f(transition_steps + k ) = y 

    Args
    ----
    x (float): initial value
    y (float): final value
    transition_steps (int): number of steps to transition over
    total_steps 

    """
    assert (total_steps >= transition_steps)
    output=np.zeros(total_steps)
    T=transition_steps-1
    output[:T] = np.array([ (T - step)/T * x + step/T * y for step in range(T)])
    output[T:] = y
    
    return output




################################################################################
print_end_of(__name__)


