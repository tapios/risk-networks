# Get the epiforecast package onto our path:
import os, sys; sys.path.append(os.path.join(".."))

import numpy as np

from epiforecast.scenarios import king_county_transition_rates

def get_rates(rates_name):
    return getattr(king_county_transition_rates(population=3, random_seed=1234), rates_name)

transition_rates = king_county_transition_rates(population=3, random_seed=1234)
 
for name in (
              'latent_periods',
              'community_infection_periods',
              'hospital_infection_periods',
              'hospitalization_fraction',
              'community_mortality_fraction',
              'hospital_mortality_fraction',
            ):

    print(get_rates(name))

#def test_latent_periods(): 
#    assert (
#            get_rates('latent_periods') 
#                == np.array([6.59337662, 6.63560047, 4.03092229])
#           ).all()

#def test_community_infection_periods(): 
#    assert (
#            get_rates('community_infection_periods') 
#                == np.array([5.7768707, 1.91422953, 7.96380771])
#           ).all()
#
#def test_hospital_infection_periods(): 
#    assert (
#            get_rates('hospital_infection_periods') 
#                == np.array([3.60376111, 8.10037068, 6.84607501])
#           ).all()
#
#def test_hospitalization_fraction(): 
#    assert (
#            get_rates('hospitalization_fraction') 
#                == np.array([3.67966623e-06, 3.78385250e-01, 4.27390020e-02])
#           ).all()
#
#def test_community_mortality_fraction(): 
#    assert (
#            get_rates('community_mortality_fraction') 
#                == np.array([5.79223912e-90, 5.03197685e-38, 0.00000000e+00])
#           ).all()
#
#def test_hospital_mortality_fraction(): 
#    assert (
#            get_rates('hospital_mortality_fraction') 
#                == np.array([[4.38776678e-76, 7.30522015e-07, 5.15894221e-02]])
#           ).all()
