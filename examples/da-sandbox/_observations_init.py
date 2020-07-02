from epiforecast.measurements import (Observation,
                                      DataObservation,
                                      HighVarianceObservation)

from _user_network_init import user_population
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
# imperfect observations #######################################################
random_infection_test = Observation(
        N=user_population,
        obs_frac=1.0,
        obs_status='I',
        obs_name="Random Infection Test",
        obs_var_min=1e-6)

high_var_infection_test = HighVarianceObservation(
        N=user_population,
        obs_frac=0.1,
        obs_status='I',
        obs_name="Test maximal variance infected",
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

