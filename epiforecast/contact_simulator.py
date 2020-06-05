import copy
import numpy as np
from numba import njit, float64
from numba.core import types
from numba.typed import Dict

from numba.experimental import jitclass

@njit
def update_common_keys(recipient, donor):
    for key in recipient.keys():
        recipient[key] = donor.get(donor[key], recipient[key])

@njit
def initialize_state(active_contacts, event_time, overshoot, edge_state):
    for i, state in enumerate(edge_state.values()):
        active_contacts[i] = state[0]
        event_time[i] = state[1]
        overshoot[i] = state[2]






class ContactSimulator:
    """
    Simulates the total contact time between people within a time interval
    using a birth/death process given a mean contact rate, and a mean contact duration.
    """

    def __init__(self, contact_network,
                                start_time = 0.0,
                       initialize_contacts = True,
                            inception_rate = None,
                       mean_event_lifetime = None):
        """
        Args
        ----
        initial_fraction_active_contacts (float): Initial fraction of active contacts between 0, 1.

        n_contacts (int): Number of contacts (default: None)

        active_contacts (np.array): Array of booleans of length `n_contacts` indicating which
                                    contacts are active (default: None)

        inception_rate (class): The average number of people each individual encounters at a time.
                                Must have a function `.rate(time)` that returns the current inception
                                rate at `time` in units of days.

        mean_event_lifetime (float): The mean duration of a contact event.

        start_time (float): the start time of the simulation in days
        """

        n_contacts = nx.number_of_edges(contact_network) # must be recalculated always
        self.contact_network = contact_network
        self.time = start_time
        self.inception_rate = inception_rate
        self.mean_event_lifetime = mean_event_lifetime

        if day_inception_rate is not None:
            nx.set_node_attributes(contact_network, values=day_inception_rate, name="day_inception_rate")

        if night_inception_rate is not None:
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
            self.active_contacts = np.zeros(n_contacts)

        try:
            self.edge_weights = nx.get_edge_attributes(self.contact_network, 'weights')
        except:
            self.edge_weights = { edge: 0.0 for edge in self.contact_network.edges() }

        self.contact_duration = np.zeros(n_contacts)
        self.overshoot_contact_duration = np.zeros(n_contacts)
        self.event_time = np.zeros(n_contacts)

        self.interval_stop_time = start_time
        self.interval_start_time = 0.0
        self.interval_steps = 0

        # We use a Gillespie algorithm based on random time steps, which means
        # we overshoot the end of an interval
        self.overshoot_time = 0.0

    def set_time(self, time):
        self.time = time
        self.interval_start_time = time - (self.interval_stop_time - self.interval_start_time)
        self.interval_stop_time = time
        self.overshoot_contact_duration *= 0

    def regenerate_edge_state(self):
        """
        Generate an empty edge state (ordered) numba dictionary corresponding to the current
        contact network.
        """
        self.edge_state = Dict.empty(
                                     key_type = types.Tuple((types.int64, types.int64)),
                                     value_type = types.Tuple((types.boolean, types.float64, types.float64))
                                    )

        self.edge_state.update({ edge: (False, 0.0, 0.0) for edge in self.contact_network.edges() })

    def run(self, stop_time):
        """
        Simulate time-dependent contacts with a birth/death process.
        """

        if stop_time <= self.interval_stop_time:
            raise ValueError("Stop time is not greater than previous interval stop time!")

        # Synchronize contact state on current edges with previous state
        self.previous_edge_state = copy.deepcopy(self.edge_state)
        self.regenerate_edge_state()
        update_common_keys(self.edge_state, self.previous_edge_state)

        n_contacts = nx.number_of_edges(self.contact_network)

        self.overshoot = np.zeros(n_contacts)
        self.active_contacts = np.zeros(n_contacts)
        self.event_time = np.zeros(n_contacts)

        initialize_state(active_contacts, event_time, overshoot, self.edge_state)

        self.contact_duration = np.zeros(n_contacts)

        # Run the simulation
        (self.time,
         self.overshoot_time,
         self.interval_steps) = gillespie_contact_simulation(self.interval_contact_duration,
                                                             self.active_contacts,
                                                             self.time,
                                                             stop_time,
                                                             self.mean_event_lifetime,
                                                             self.inception_rate)

        # Store the "overshoot contact duration" for use during a subsequent simulation.
        self.overshoot_contact_duration = self.overshoot_time * self.active_contacts

        # Record the start and stop times of the current simulation interval
        self.interval_start_time = self.interval_stop_time
        self.interval_stop_time = stop_time

    def mean_contact_duration(self, stop_time, **kwargs):
        self.run(stop_time, **kwargs)
        time_interval = self.interval_stop_time - self.interval_start_time
        return self.interval_contact_duration / time_interval




specification = [
                 ('maximum', float64),
                 ('minimum', float64)
                ]

@jitclass(specification)
class DiurnalContactInceptionRate:
    def __init__(self, maximum=22, minimum=3):
        self.maximum = maximum
        self.minimum = minimum

    def rate(self, t):
        return np.maximum(self.minimum, self.maximum * (1 - np.cos(np.pi * t)**4)**4)
