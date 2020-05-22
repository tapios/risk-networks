import numpy as np


class MasterEquationModelEnsemble:
    def __init__(self,
            ensemble_size = 1,
            contact_network = contact_network,
            transmission_rates = constant_transmission_rate,
            state_transition_rates = transition_rates):
        pass

    #  Setter methods ----------------------------------------------------------
    def set_state(self):
        # Set the master equation state to the same initial state as the kinetic model.
        # master_model.set_state(initial_state)
        pass

    def set_contact_network(self):
        pass

    def set_ensemble(self, new_ensemble):
        pass

    def set_transition_rates(self, new_transition_rates):
        pass

    def set_transmission_rates(self, new_transmission_rates):
        pass

    def populate_states(self, distribution=state_distribution):
        pass

    def populate_transition_rates(self, distribution=transition_rates_distribution):
        pass

    def populate_transmission_rates(self, distribution=transmission_rates_distribution):
        pass

    # Ode solver methods -------------------------------------------------------
    def simulate(self, ):
        # Simulate an epidemic over the static_contacts_interval.
        # output = master_model.simulate(static_contacts_interval)
        pass
