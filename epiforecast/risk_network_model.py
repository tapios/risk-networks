import numpy as np
import scipy.sparse as sps
import networkx as nx

def trivial_closure(infection_pressure): return None

class StaticRiskNetworkModel:
    """A statistical epidemic model for the expected status of individuals
    on a static contact network."""
    def __init__(self, 
                                         N_nodes = 1, 
                                 contact_network = None, 
                                transition_rates = None, 
                      infection_pressure_closure = trivial_closure
                 ):

        self.N_nodes = N_nodes

        # This method is used during time-integration to 'add' a closure term to the infection pressure
        # experienced by each node.
        self.add_infection_pressure_closure = infection_pressure_closure

        if contact_network is not None: self.set_contact_network(contact_network)
        if transition_rates is not None: self.set_transition_rates(transition_rates)

    def set_transition_rates(self, transition_rates):
        """Set transition rates between the Exposed, Infected, Hospitalized, Resistant, and Deceased states."""

        self.transition_rates = transition_rates

        # Unpack and broadcast to arrays of length N_nodes
        sigma       = np.ones(self.N_nodes) * transition_rates.exposed_to_infected
        delta       = np.ones(self.N_nodes) * transition_rates.infected_to_hospitalized
        theta       = np.ones(self.N_nodes) * transition_rates.infected_to_resistant
        mu          = np.ones(self.N_nodes) * transition_rates.infected_to_deceased
        theta_prime = np.ones(self.N_nodes) * transition_rates.hospitalized_to_resistant
        mu_prime    = np.ones(self.N_nodes) * transition_rates.hospitalized_to_deceased

        # Build the matrix of transition rates to be used in forward and backward solves.
        self.transition_rates_matrix = sps.csr_matrix(sps.bmat([

            [sps.eye(self.N_nodes), None,              None,              None,                    None, None],
            [sps.eye(self.N_nodes), sps.diags(-sigma), None,              None,                    None, None],
            [None,                  sps.diags(sigma),  sps.diags(-gamma), None,                    None, None],
            [None,                  None,              sps.diags(delta),  sps.diags(-gamma_prime), None, None],
            [None,                  None,              sps.diags(theta),  sps.diags(theta_prime),  None, None],
            [None,                  None,              sps.diags(mu),     sps.diags(mu_prime),     None, None]

        ], format = 'csr'), shape = [6 * self.N_nodes, 6 * self.N_nodes])

        self.transmission_matrix = np.array([transition_rates.beta_ij, self.beta_prime_ij])

    def set_contact_network(self, contact_network):
        """Set the contact network and convert contact network to a scipy sparse matrix."""

        self.contact_network = contact_network
        self.contact_matrix = nx.to_scipy_sparse_matrix(self.contact_network)

    def calculate_forwards_tendency(self, time, state):
        """Calculate + d/dt state. This function is used by the scipy integrator."""

        iS, iE, iI, iH = [range(j * self.N_nodes, (j + 1) * self.N_nodes) for j in range(4)]

        self.infection_pressure = sps.kron(self.transmission_matrix, 
                                           self.contact_matrix).dot(np.hstack([state[iI], state[iH]]))

        self.add_infection_pressure_closure(self.infection_pressure)

        self.transition_rates_matrix[iS, iS] = - self.infection_pressure
        self.transition_rates_matrix[iE, iS] =   self.infection_pressure

        dstates_dt = self.transition_rates_matrix.dot(state)

        return dstates_dt

    def calculate_backwards_tendency(self, time, state):
        """Calculate - d/dt state. This function is used by the scipy integrator."""

        iS, iE, iI, iH = [range(j * self.N_nodes, (j + 1) * self.N_nodes) for j in range(4)]

        self.infection_pressure = sps.kron(self.transmission_matrix, 
                                           self.contact_matrix).dot(np.hstack([state[iI], state[iH]]))

        self.add_infection_pressure_closure(self.infection_pressure)

        self.transition_rates_matrix[iS, iS] = - self.infection_pressure
        self.transition_rates_matrix[iE, iS] =   self.infection_pressure

        dstates_dt = self.transition_rates_matrix.dot(state)

        return - dstates_dt

    def run_forwards(self, initial_state, **kwargs):
        forwards_tendency = lambda time, state: self.calculate_forwards_tendency(time, state)
        return scipy.integrate.solve_ivp(forwards_tendency,
                                         y0 = initial_state, **kwargs)

    def run_backwards(self, initial_state, **kwargs):
        backwards_tendency = lambda time, state: self.calculate_backwards_tendency(time, state)
        return scipy.integrate.solve_ivp(backwards_tendency,
                                         y0 = initial_state, **kwargs)


class TransitionRates:
    """A container for transition rates. """
    def __init__(self, 
                 # TODO: fill in sensible defaults
                               transmission_rate = 0, # sigma
                      hospital_transmission_rate = 0, # sigma
                             exposed_to_infected = 0, # sigma
                        infected_to_hospitalized = 0, # h * gamma
                           infected_to_resistant = 0, # (1 - h - d) * gamma
                            infected_to_deceased = 0, # d * gamma
                       hospitalized_to_resistant = 0, # (1 - d_prime ) * gamma_prime
                        hospitalized_to_deceased = 0, # d_prime * gamma_prime
                 ):
                       
        self.transmission_rate          = self.beta_ij       = transmission_rate
        self.hospital_transmission_rate = self.beta_prime_ij = hospital_transmission_rate
        self.exposed_to_infected        = self.sigma         = exposed_to_infected
        self.infected_to_hospitalized   = self.delta         = infected_to_hospitalized
        self.infected_to_resistant      = self.theta         = infected_to_resistant
        self.infected_to_deceased       = self.mu            = infected_to_deceased
        self.hospitalized_to_resistant  = self.theta_prime   = hospitalized_to_resistant
        self.hospitalized_to_deceased   = self.mu_prime      = hospitalized_to_deceased
