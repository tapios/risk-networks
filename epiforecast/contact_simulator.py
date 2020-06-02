import numpy as np
from numba import njit, float64
from numba.experimental import jitclass

@njit
def accumulate_contact_duration(contact_duration, time_step, active_contacts):
    contact_duration += time_step * active_contacts

@njit
def random_index_to_deactivate(active_contacts, n_active_contacts):
    k = np.random.choice(n_active_contacts) # select an active contact
    i = np.where(active_contacts)[0][k]
    return i 

@njit
def random_index_to_activate(active_contacts, n_active_contacts):
    k = np.random.choice(len(active_contacts) - n_active_contacts) # select an inactive contact
    i = np.where(~active_contacts)[0][k]
    return i

@njit
def gillespie_step(contact_duration, active_contacts, time, stop_time,
                   mean_event_lifetime, current_inception_rate):

    n_total = len(active_contacts)
    n_active = np.count_nonzero(active_contacts)

    activation_rate = current_inception_rate * (n_total - n_active)
    deactivation_rate = n_active / mean_event_lifetime
     
    # Generate exponentially-distributed random time step:
    time_step = -np.log(np.random.random()) / (deactivation_rate + activation_rate)

    if time + time_step > stop_time: # event occurs after the end of the simulation
        # Add only the part of the contact duration that occurs within the current interval
        accumulate_contact_duration(contact_duration, stop_time - time, active_contacts)

    else: # event occurs within the simulation interval
        accumulate_contact_duration(contact_duration, time_step, active_contacts)
                                    
    # Because a random, exponentially-distributed amount of time has elapsed, 
    # an "event" occurs (of course)! The event is the activation or deactivation
    # of a contact.
    deactivation_probability = deactivation_rate / (deactivation_rate + activation_rate)

    # Draw from uniform random distribution on [0, 1) to decide
    # whether to activate or deactivate contacts
    if np.random.random() < deactivation_probability:
        i = random_index_to_deactivate(active_contacts, n_active)
        active_contacts[i] = False
    else: 
        i = random_index_to_activate(active_contacts, n_active)
        active_contacts[i] = True

    # Becomes positive when time + time_step exceeds the stop_time
    overshoot_time = time + time_step - stop_time

    return time_step, overshoot_time


@njit
def gillespie_contact_simulation(contact_duration, active_contacts, time, stop_time,
                                 mean_event_lifetime, inception):

    interval_steps = 0 # bookkeeping

    while time < stop_time:
            
        current_inception_rate = inception.rate(time)

        time_step, overshoot_time = gillespie_step(contact_duration,
                                                   active_contacts,
                                                   time,
                                                   stop_time,
                                                   mean_event_lifetime,
                                                   current_inception_rate)

        # Move into the future
        time += time_step
        interval_steps += 1

    return time, overshoot_time, interval_steps




class ContactSimulator:
    """
    Simulates the total contact time between people within a time interval
    using a birth/death process given a mean contact rate, and a mean contact duration.
    """

    def __init__(self, initial_fraction_active_contacts = 0.0,
                                             start_time = 0.0,
                                             n_contacts = None, 
                                        active_contacts = None,
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

        self.time = start_time
        self.inception_rate = inception_rate
        self.mean_event_lifetime = mean_event_lifetime
        
        # Initialize the active contacts
        if active_contacts is not None: # list of active contacts is provided
            self.active_contacts = active_contacts
        
        else: # generate active contacts randomly

            if n_contacts is None: 
                raise ValueError("n_contacts must be specified if active_contacts is not supplied!")

            # If contact information is supplied, use equilibrium solution to determine
            # the initial fraction of active contacts.
            if inception_rate is not None and mean_event_lifetime is not None:  

                λ = inception_rate.rate(start_time) # current mean contact rate
                μ = 1 / mean_event_lifetime
                initial_fraction_active_contacts = λ / (μ + λ)

            # Otherwise, initial_fraction_active_contacts is left unchanged
            active_probability = initial_fraction_active_contacts
            inactive_probability = 1 - active_probability

            self.active_contacts = np.random.choice([False, True],
                                                    size = n_contacts, 
                                                    p = [inactive_probability, active_probability])

        # Initialize array of contact durations within a simulation interval
        self.n_contacts = len(self.active_contacts)
        self.interval_contact_duration = np.zeros(self.n_contacts)
        self.interval_stop_time = start_time
        self.interval_start_time = 0.0
        self.interval_steps = 0

        # We use a Gillespie algorithm based on random time steps, which means
        # we overshoot the end of an interval
        self.overshoot_contact_duration = np.zeros(self.n_contacts)
        self.overshoot_time = 0.0
        
    def set_time(self, time):
        self.time = time
        self.interval_start_time = time - (self.interval_stop_time - self.interval_start_time)
        self.interval_stop_time = time
        self.overshoot_contact_duration *= 0

    def run(self, stop_time, mean_event_lifetime=None, inception_rate=None):
        """
        Simulate time-dependent contacts with a birth/death process.
        """

        if stop_time <= self.interval_stop_time:
            raise ValueError("Stop time is not greater than previous interval stop time!")

        if mean_event_lifetime is not None:
            self.mean_event_lifetime = mean_event_lifetime

        if inception_rate is not None:
            self.inception_rate = inception_rate

        if self.mean_event_lifetime is None:
            raise ValueError("Mean event duration is not set!")

        if self.inception_rate is None:
            raise ValueError("Mean contact rate is not set!")

        # Capture the contact duration associated with 'overshoot' during a previous simulation:
        if stop_time > self.time:
            self.interval_contact_duration = self.overshoot_contact_duration

        else: # stop_time is within the current event interval; no events will occur.
            self.interval_contact_duration = (stop_time - self.interval_stop_time) * self.active_contacts
            overshoot_time = self.time - stop_time

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
