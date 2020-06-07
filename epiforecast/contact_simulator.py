import copy
import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit, prange, float64
import networkx as nx

@njit
def normalize(edge):
    n, m = edge
    if m > n: # switch
        n, m = m, n
    return n, m

@njit
def initialize_state(active_contacts, event_time, overshoot_duration, edge_state):
    """
    Initialize active_contacts, event_time, overshoot_duration to the state in edge_state.
    """
    for i, state in enumerate(edge_state.values()):
        active_contacts[i] = state[0]
        event_time[i] = state[1]
        overshoot_duration[i] = state[2]

@njit
def synchronize_state(edge_state, active_contacts, event_time, overshoot_duration):
    """
    Synchronize the edge_state with active_contacts, event_time, and the overshoot_duration.
    """
    for i, edge in enumerate(edge_state):
        edge_state[edge] = (active_contacts[i], event_time[i], overshoot_duration[i])

def remove_edges(edge_state, edge_list):
    """
    Delete edges in edge_list from edge_state.
    """
    if len(edge_list) > 0:
        for edge in edge_list:
            try:
                del edge_state[edge]
            except:
                print(edge, "not found in edge_state, perhaps because",
                      edge[0], "and", edge[1], "are both hospitalized.")
                pass

def add_edges(edge_state, edge_list):
    """
    Add edges from edge_list to edge_state with default initial values.
    """
    if len(edge_list) > 0:
        edge_state.update({ edge: (False, 0.0, 0.0) for edge in map(normalize, edge_list) })

@njit
def calculate_inception_rates(day_inception_rate, night_inception_rate,
                              nodal_day_inception_rate, nodal_night_inception_rate,
                              edge_state):
    """
    Calculate the inspection rates for each contact given the settings for each node.
    """

    for i, edge in enumerate(edge_state):

        day_inception_rate[i] = np.minimum(nodal_day_inception_rate[edge[0]],
                                           nodal_day_inception_rate[edge[1]])

        night_inception_rate[i] = np.minimum(nodal_night_inception_rate[edge[0]],
                                             nodal_night_inception_rate[edge[1]])

def generate_edge_state(contact_network):
    """
    Generate an empty edge state (ordered) numba dictionary corresponding to the current
    contact network.
    """
    edge_state = Dict.empty(key_type = types.Tuple((types.int64, types.int64)),
                            value_type = types.Tuple((types.boolean, types.float64, types.float64)))

    # Takes 19 s for 100000
    edge_state.update({ edge: (False, 0.0, 0.0) for edge in map(normalize, contact_network.edges()) })

    # Takes 6 s for 100000? But returns a python rather than a numba dictionary.
    #edge_state = { edge: (False, 0.0, 0.0) for edge in contact_network.edges() }

    return edge_state

class ContactSimulator:
    """
    Simulates the total contact time between people within a time interval
    using a birth/death process given a mean contact rate, and a mean contact duration.
    """

    def __init__(self, contact_network,
                       initialize_contacts = True,
                        day_inception_rate = None,
                      night_inception_rate = None,
                       mean_event_lifetime = None,
                                start_time = 0.0,
                           contacts_buffer = 0,
                   rate_integral_increment = 0.05):
        """
        Args
        ----
        initialize_contacts (bool): Whether or not to initialize the contact activity.

        day_inception_rate (float): The rate of inception of new contacts during the day.

        night_inception_rate (float): The rate of inception of new contacts at night.

        mean_event_lifetime (float): The mean duration of a contact event.

        start_time (float): Start time of the simulation, in days.

        contacts_buffer (int): The size of the buffer to keep in state arrays in the
                               case that new edges are added.

        rate_integral_increment (float): Increment used to integrate the time-varying inception rate
                                         within the Gillespie simulation of contact activity.
                                         Must be much shorter than one day when the inception rate varies
                                         diurnally.
        """

        # Number of possible edges during an epidemic simulation
        n_contacts = nx.number_of_edges(contact_network) + contacts_buffer

        self.contact_network = contact_network
        self.edge_state = generate_edge_state(contact_network)
        self.time = start_time
        self.mean_event_lifetime = mean_event_lifetime
        self.rate_integral_increment = rate_integral_increment

        if day_inception_rate is not None: # set day inception rates on the contact network nodes
            nx.set_node_attributes(contact_network, values=day_inception_rate, name="day_inception_rate")

        if night_inception_rate is not None: # set night inception rates contact network nodes
            nx.set_node_attributes(contact_network, values=night_inception_rate, name="night_inception_rate")

        # Initialize the active contacts
        if initialize_contacts:
            λ = night_inception_rate
            μ = 1 / mean_event_lifetime
            initial_fraction_active_contacts = λ / (μ + λ)

            # Otherwise, initial_fraction_active_contacts is left unchanged
            active_probability = initial_fraction_active_contacts
            inactive_probability = 1 - active_probability

            self.active_contacts = np.random.choice([False, True],
                                                    size = n_contacts,
                                                    p = [inactive_probability, active_probability])

        else:
            self.active_contacts = np.zeros(n_contacts, dtype=bool)

        self.contact_duration = np.zeros(n_contacts)
        self.overshoot_duration = np.zeros(n_contacts)
        self.event_time = np.zeros(n_contacts)

        self.day_inception_rate = day_inception_rate * np.ones(n_contacts)
        self.night_inception_rate = night_inception_rate * np.ones(n_contacts)

        self.interval_stop_time = start_time
        self.interval_start_time = 0.0

    def run(self, stop_time, admitted_patients=[], discharged_patients=[]):
        """
        Simulate time-dependent contacts with a birth/death process.
        """

        if stop_time <= self.interval_stop_time:
            raise ValueError("Stop time is not greater than previous interval stop time!")

        # Determine the number of contacts on the current contact network
        n_contacts = nx.number_of_edges(self.contact_network)

        # Fiddle with the contact state if edges have been added or deleted
        if len(admitted_patients) != 0 or len(discharged_patients) != 0:

            # First, synchronize current edge state prior to hospitalization
            synchronize_state(self.edge_state, self.active_contacts, self.event_time, self.overshoot_duration)

            # Find edges to add and remove
            edges_to_remove = set()
            edges_to_add = set()

            if len(admitted_patients) != 0:
                for patient in admitted_patients:
                    edges_to_remove.update(map(normalize, patient.community_contacts))
                    edges_to_add.update(map(normalize, patient.health_worker_contacts))

            if len(discharged_patients) != 0:
                for patient in discharged_patients:
                    edges_to_remove.update(map(normalize, patient.health_worker_contacts))
                    edges_to_add.update(map(normalize, patient.community_contacts))

            remove_edges(self.edge_state, edges_to_remove)
            add_edges(self.edge_state, edges_to_add)

            # Reinitialize the state with the new edge list
            initialize_state(self.active_contacts, self.event_time, self.overshoot_duration, self.edge_state)

        # Re-estimate day_inception_rate and night_inception_rate
        nodal_day_inception_rate = np.array(
            [ data['day_inception_rate'] for node, data in self.contact_network.nodes(data=True) ])

        nodal_night_inception_rate = np.array(
            [ data['night_inception_rate'] for node, data in self.contact_network.nodes(data=True) ])

        day_inception_rate = np.zeros(n_contacts)
        night_inception_rate = np.zeros(n_contacts)

        calculate_inception_rates(self.day_inception_rate, self.night_inception_rate,
                                  nodal_day_inception_rate, nodal_night_inception_rate,
                                  self.edge_state)

        # Simulate contacts using a time dependent gillespie algorithm for a birth death process
        # with varying birth rate.
        simulate_contacts(
                          n_contacts,
                          stop_time,
                          self.event_time,
                          self.contact_duration,
                          self.overshoot_duration,
                          self.active_contacts,
                          self.night_inception_rate,
                          self.day_inception_rate,
                          self.mean_event_lifetime,
                          self.rate_integral_increment
                         )

        # Record the start and stop times of the current simulation interval
        self.interval_start_time = self.interval_stop_time
        self.interval_stop_time = stop_time

    def run_and_set_edge_weights(self, stop_time, **kwargs):
        self.run(stop_time, **kwargs)
        time_interval = self.interval_stop_time - self.interval_start_time
        edge_weights = { edge: self.contact_duration[i] / time_interval
                         for i, edge in enumerate(self.edge_state) }

        nx.set_edge_attributes(self.contact_network, values=edge_weights, name='exposed_by_infected')
        nx.set_edge_attributes(self.contact_network, values=edge_weights, name='exposed_by_hospitalized')

    def set_time(self, time):
        self.time = time
        self.interval_start_time = time - (self.interval_stop_time - self.interval_start_time)
        self.interval_stop_time = time
        self.overshoot_duration *= 0


# For implementation of Gillespie simulation with time-dependent rates, see discussion in
#
# Christian L. Vestergaard , Mathieu Génois, "Temporal Gillespie Algorithm: Fast Simulation
# of Contagion Processes on Time-Varying Networks", PLOS Computational Biology (2015)
#
# https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004579

@njit
def diurnal_inception_rate(λnight, λday, t):
    return np.maximum(λnight, λday * (1 - np.cos(np.pi * t)**4)**4)

@njit(parallel=True)
def simulate_contacts(
                      n_contacts,
                      stop_time,
                      event_time,
                      contact_duration,
                      overshoot_duration,
                      active_contacts,
                      night_inception_rate,
                      day_inception_rate,
                      mean_event_lifetime,
                      rate_integral_increment
                     ):

    for i in prange(n_contacts):

        (event_time[i],
         active_contacts[i],
         contact_duration[i],
         overshoot_duration[i]) = simulate_contact(
                                                   stop_time,
                                                   event_time[i],
                                                   contact_duration[i],
                                                   overshoot_duration[i],
                                                   active_contacts[i],
                                                   night_inception_rate[i],
                                                   day_inception_rate[i],
                                                   mean_event_lifetime,
                                                   rate_integral_increment
                                                  )

@njit
def simulate_contact(
                     stop_time,
                     event_time,
                     contact_duration,
                     overshoot_duration,
                     active_contact,
                     night_inception_rate,
                     day_inception_rate,
                     mean_event_lifetime,
                     rate_integral_increment
                    ):

    contact_duration = overshoot_duration

    while event_time < stop_time:

        if active_contact: # Compute contact deactivation time.

            # Contact is deactivated after
            time_step = - np.log(np.random.random()) * mean_event_lifetime
            contact_duration += time_step

            # Deactivation
            event_time += time_step
            active_contact = False

        else: # Compute next contact inception.

            # Normalized step with τ ~ Exp(1)
            τ = - np.log(np.random.random())

            # Solve
            #
            #       τ = ∫ λ(t) dt
            #
            # where the integral goes from the current time to the time of the
            # next event, or from t to t + Δt, where Δt = time_step.
            #
            # For this we accumulate the integral Λ = ∫ λ dt using trapezoidal integration
            # over microintervals of length δ. When Λ > τ, we calculate Δt with an
            # O(δ) approximation. An O(δ²) approximation is also possible, but we do not
            # pursue this here.
            #
            # Definitions:
            #   - δ: increment width
            #   - n: increment counter
            #   - λⁿ: λ at t = event_time + n * δ
            #   - λᵐ: λ at t = event_time + m * δ, with m = n-1
            #   - Λⁿ: integral of λ(t) from event_time to n * δ

            δ = rate_integral_increment  # 0.05 # day
            n = 1
            λᵐ = diurnal_inception_rate(night_inception_rate, day_inception_rate, event_time)
            λⁿ = diurnal_inception_rate(night_inception_rate, day_inception_rate, event_time + δ)
            Λⁿ = δ / 2 * (λᵐ + λⁿ)

            while Λⁿ < τ:
                n += 1
                λᵐ = λⁿ
                λⁿ = diurnal_inception_rate(night_inception_rate, day_inception_rate, event_time + n * δ)
                Λⁿ += δ / 2 * (λᵐ + λⁿ)

            # O(δ²) approximation for time_step.
            time_step = n * δ - (Λⁿ - τ) * 2 / (λⁿ + λᵐ)

            # Contact inception
            event_time += time_step
            active_contact = True

    # We 'overshoot' the end of the interval. To account for this, we subtract the
    # overshoot, and the contribution of the 'overshoot' to the total contact duration.
    #
    #              < -------------------- >
    #                     overshoot
    #
    #             stop
    # ----- x ---- | -------------------- x
    #     prev                          next
    #     step                          step
    #
    # A confusing part of this algorithm is that the *current* state is the inverse
    # of the prior state. Therefore if we are *currently* in contact, we were *not*
    # in contact during the overshoot interval --- and vice versa. This is why
    # we write (1 - contact) below.

    overshoot_duration = (event_time - stop_time) * (1 - active_contact)
    contact_duration -= overshoot_duration

    return event_time, active_contact, contact_duration, overshoot_duration
