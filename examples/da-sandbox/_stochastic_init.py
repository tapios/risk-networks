from epiforecast.populations import TransitionRates
from epiforecast.health_service import HealthService
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
from epiforecast.scenarios import random_epidemic

from _constants import (latent_periods,
                        community_infection_periods,
                        hospital_infection_periods,
                        hospitalization_fraction,
                        community_mortality_fraction,
                        hospital_mortality_fraction,
                        community_transmission_rate,
                        hospital_transmission_reduction,
                        static_contact_interval,
                        mean_contact_lifetime,
                        min_contact_rate,
                        max_contact_rate,
                        start_time,
                        SEED_STOCHASTIC_INIT_1,
                        SEED_STOCHASTIC_INIT_2,
                        SEED_STOCHASTIC_INIT_3)
from _network_init import network, population, populace
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
# transition rates a.k.a. independent rates (σ, γ etc.) ########################
# constructor takes clinical parameter samplers which are then used to draw real
# clinical parameters, and those are used to calculate transition rates
transition_rates = TransitionRates.from_samplers(
        population=network.get_node_count(),
        lp_sampler=latent_periods,
        cip_sampler=community_infection_periods,
        hip_sampler=hospital_infection_periods,
        hf_sampler=hospitalization_fraction,
        cmf_sampler=community_mortality_fraction,
        hmf_sampler=hospital_mortality_fraction,
        distributional_parameters=network.get_age_groups()
)

transition_rates.calculate_from_clinical()
network.set_transition_rates_for_kinetic_model(transition_rates)

# health service ###############################################################
health_service = HealthService(original_contact_network=network,
                               health_workers=network.get_health_workers(),
                               seed=SEED_STOCHASTIC_INIT_1)

# epidemic simulator ###########################################################
epidemic_simulator = EpidemicSimulator(
        network,
        community_transmission_rate,
        hospital_transmission_reduction,
        static_contact_interval,
        mean_contact_lifetime,
        day_inception_rate=max_contact_rate,
        night_inception_rate=min_contact_rate,
        health_service=health_service,
        start_time=start_time,
        seed=SEED_STOCHASTIC_INIT_2)

kinetic_ic = random_epidemic(population,
                             populace,
                             fraction_infected=0.01,
                             seed=SEED_STOCHASTIC_INIT_3)
epidemic_simulator.set_statuses(kinetic_ic)

################################################################################
print_end_of(__name__)


