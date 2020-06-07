import numpy as np
import networkx as nx
from .simulation import Gillespie_simple_contagion

def print_initial_statuses(statuses,population):
    #use default dict here
    for i in range(population-1):
        print(statuses[i], end=" ")

    print("")

def print_statuses(statuses):

    for i in sorted(list(statuses.keys())):
        print(statuses[i], end=" ")

    print("")


class KineticModel:
    def __init__(self,
                 contact_network,
                 transition_rates,
                 community_transmission_rate,
                 hospital_transmission_reduction,
                 mean_contact_duration = None,
                 start_time = 0.0,
                 ):
        """
        A class to implement a Kinetic Monte-Carlo solver on a provided network.

        Args
        -----
        contact_network (networkx.Graph): The contact network

        mean_contact_duration (np.array): Mean contact duration of each node in the static
                                          network over which we simulate

        transition_rates (TransitionRates): Contains all transition rates as dictionaries of size
                                            of health worker + community population
                                            for example: transition_rates.susceptible_to_exposed

        community_transmission_rate (float): Global constant transmission rate for the community(often referred within as beta)

        hospital_transmission_reduction (float): reduction factor for the transmission rate for those in hospital
                                                 hospital_transmission_rate = community transmission_rate *
                                                                              hospital_transmission_reduction

        """

        self.contact_network = contact_network
        self.community_transmission_rate = community_transmission_rate
        self.hospital_transmission_rate = community_transmission_rate * hospital_transmission_reduction

        #if mean_contact_duration is None:
        #    mean_contact_duration = np.zeros(nx.number_of_edges(contact_network))

        #self.set_mean_contact_duration(mean_contact_duration)

        # What statuses to return from Gillespie simulation
        self.return_statuses = ('S', 'E', 'I', 'H', 'R', 'D')

        # Independent rates diagram
        self.diagram_indep = nx.DiGraph()
        self.diagram_indep.add_node('S')
        self.diagram_indep.add_edge('E', 'I', rate=1, weight_label='exposed_to_infected')
        self.diagram_indep.add_edge('I', 'H', rate=1, weight_label='infected_to_hospitalized')
        self.diagram_indep.add_edge('I', 'R', rate=1, weight_label='infected_to_resistant')
        self.diagram_indep.add_edge('I', 'D', rate=1, weight_label='infected_to_deceased')
        self.diagram_indep.add_edge('H', 'R', rate=1, weight_label='hospitalized_to_resistant')
        self.diagram_indep.add_edge('H', 'D', rate=1, weight_label='hospitalized_to_deceased')

        # Neighbor-induced rates diagram
        self.diagram_neigh = nx.DiGraph()
        self.diagram_neigh.add_edge(('I', 'S'), ('I', 'E'), rate = self.community_transmission_rate, weight_label = 'exposed_by_infected')
        self.diagram_neigh.add_edge(('H', 'S'), ('H', 'E'), rate = self.hospital_transmission_rate,  weight_label = 'exposed_by_hospitalized')

        # Set the transition rates:
        nx.set_node_attributes(self.contact_network, values=transition_rates.exposed_to_infected,       name='exposed_to_infected')
        nx.set_node_attributes(self.contact_network, values=transition_rates.infected_to_hospitalized,  name='infected_to_hospitalized')
        nx.set_node_attributes(self.contact_network, values=transition_rates.infected_to_resistant,     name='infected_to_resistant')
        nx.set_node_attributes(self.contact_network, values=transition_rates.infected_to_deceased,      name='infected_to_deceased')
        nx.set_node_attributes(self.contact_network, values=transition_rates.hospitalized_to_resistant, name='hospitalized_to_resistant')
        nx.set_node_attributes(self.contact_network, values=transition_rates.hospitalized_to_deceased,  name='hospitalized_to_deceased')

        self.current_time = start_time
        self.current_statuses = None # must be set by set_statuses

        self.times = []
        self.statuses = {s: [] for s in self.return_statuses}

    def set_contact_network(self, contact_network):
        # Note: we only modify edges of the network, thus the transition rates do not need updating here.
        self.contact_network = contact_network

    def set_mean_contact_duration(self, mean_contact_duration):
        """
        Set the weights of self.contact_network, which correspond to the mean contact
        duration over a given time interval.

        Args
        ----

        mean_contact_duration (np.array) : np.array of the contact duration for each edge
                                           in the contact network.
        """
        weights = {tuple(edge): mean_contact_duration[i]
                   for i, edge in enumerate(nx.edges(self.contact_network))}

        nx.set_edge_attributes(self.contact_network, values=weights, name='exposed_by_infected')
        nx.set_edge_attributes(self.contact_network, values=weights, name='exposed_by_hospitalized')

    def set_statuses(self, statuses):
        self.current_statuses = statuses

    def simulate(self, time_interval, initial_statuses=None):
        """
        Runs the Gillespie solver with our given graph with current contact network.

        Args
        ----

        time_interval (float) : the integration time (over a static contact network)

        initial_statuses (dict) : a {node number : node status} dictionary ; the initial condition for the solve step
        """

        if initial_statuses is None:
            initial_statuses = self.current_statuses

        res = Gillespie_simple_contagion(self.contact_network,
                                         self.diagram_indep,
                                         self.diagram_neigh,
                                         initial_statuses,
                                         self.return_statuses,
                                         return_full_data = True,
                                         tmin = self.current_time,
                                         tmax = self.current_time + time_interval)

        new_times, new_statuses = res.summary()

        self.current_time += time_interval

        self.times.extend(new_times)

        for s in self.return_statuses:
            self.statuses[s].extend(new_statuses[s])

        self.current_statuses = res.get_statuses(time=self.current_time)

        return self.current_statuses
