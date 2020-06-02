import numpy as np

def accumulate_contact_duration(contact_duration, time_step, active_contacts):
    contact_duration += time_step * active_contacts

def random_index_to_deactivate(active_contacts, n_active_contacts):
    k = np.random.choice(n_active_contacts) # select an active contact
    i = np.where(active_contacts)[0][k]
    return i 

def random_index_to_activate(active_contacts, n_active_contacts):
    k = np.random.choice(len(active_contacts) - n_active_contacts) # select an inactive contact
    i = np.where(~active_contacts)[0][k]
    return i

def gillespie_step(contact_duration, active_contacts, mean_event_duration, mean_contact_rate, 
                   time, stop_time, overshoot_contact_duration, overshoot_time):

    n_total = len(active_contacts)
    n_active = np.count_nonzero(active_contacts)

    activation_rate = mean_contact_rate * (n_total - n_active)
    deactivation_rate = n_active / mean_event_duration
     
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






class ContactSimulator:
    """
    Simulates the total contact time between people within a time interval
    using a birth/death process given a mean contact rate, and a mean contact duration.
    """

    def __init__(self, initial_fraction_active_contacts = 0.0,
                                             start_time = 0.0,
                                             n_contacts = None, 
                                        active_contacts = None,
                                      mean_contact_rate = None,
                                    mean_event_duration = None,
                ):
        """
        Args
        ----
        initial_fraction_active_contacts (float): Initial fraction of active contacts between 0, 1.

        n_contacts (int): Number of contacts (default: None)

        active_contacts (np.array): Array of booleans of length `n_contacts` indicating which
                                    contacts are active (default: None)

        mean_contact_rate (callable): The average number of people each individual encounters at a time.
                                      Must be callable with `time` (with units of days) as an argument.

        mean_event_duration (float): The mean duration of a contact event.

        start_time (float): the start time of the simulation in days
        """

        self.time = start_time
        self.mean_contact_rate = mean_contact_rate
        self.mean_event_duration = mean_event_duration
        
        # Initialize the active contacts
        if active_contacts is not None: # list of active contacts is provided
            self.active_contacts = active_contacts
        
        else: # generate active contacts randomly

            if n_contacts is None: 
                raise ValueError("n_contacts must be specified if active_contacts is not supplied!")

            # If contact information is supplied, use equilibrium solution to determine
            # the initial fraction of active contacts.
            if mean_contact_rate is not None and mean_event_duration is not None:  

                λ = mean_contact_rate(start_time) # current mean contact rate
                μ = 1 / mean_event_duration
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
        
    def _initialize_run(self, stop_time, mean_event_duration, mean_contact_rate):
        """
        Initialize a forward run of the ContactSimulator
        """

        if stop_time <= self.interval_stop_time:
            raise ValueError("Stop time is not greater than previous interval stop time!")

        if mean_event_duration is not None:
            self.mean_event_duration = mean_event_duration

        if mean_contact_rate is not None:
            self.mean_contact_rate = mean_contact_rate

        if self.mean_event_duration is None:
            raise ValueError("Mean event duration is not set!")

        if self.mean_contact_rate is None:
            raise ValueError("Mean contact rate is not set!")

        # Capture the contact duration associated with 'overshoot' during a previous simulation:
        if stop_time > self.time:
            self.interval_contact_duration = self.overshoot_contact_duration
            overshoot_time = 0.0

        else: # stop_time is within the current event interval; no events will occur.
            self.interval_contact_duration = (stop_time - self.interval_stop_time) * self.active_contacts
            overshoot_time = self.time - stop_time

        self.interval_steps = 0 # bookkeeping

        return overshoot_time

    def run(self, stop_time, mean_event_duration=None, mean_contact_rate=None):
        """
        Simulate time-dependent contacts with a birth/death process.

        Args
        ----

        stop_time (float): Time to stop running the Gillespie simulation of contacts.

        mean_event_duration (float): The mean duration of a "contact event"

        mean_contact_rate (callable): A function that returns mean_contact_rate(epidemic_day).
        """

        overshoot_time = self._initialize_run(stop_time, mean_event_duration, mean_contact_rate)
    
        # Run:
        while self.time < stop_time:
            
            current_mean_contact_rate = self.mean_contact_rate(self.time)

            time_step, overshoot_time = gillespie_step(self.interval_contact_duration,
                                                       self.active_contacts,
                                                       self.mean_event_duration,
                                                       current_mean_contact_rate,
                                                       self.time,
                                                       stop_time,
                                                       self.overshoot_contact_duration,
                                                       self.overshoot_time)

            # Move into the future
            self.time += time_step
            self.interval_steps += 1

        # Store the "overshoot contact duration" for use during a subsequent simulation.
        self.overshoot_time = overshoot_time
        self.overshoot_contact_duration = overshoot_time * self.active_contacts

        # Record the start and stop times of the current simulation interval
        self.interval_start_time = self.interval_stop_time
        self.interval_stop_time = stop_time

    def mean_contact_duration(self, stop_time, **kwargs):
        self.run(stop_time, **kwargs)
        time_interval = self.interval_stop_time - self.interval_start_time
        return self.interval_contact_duration / time_interval



def cos4_diurnal_modulation(t, cmin_i, cmax_i, cmin_j, cmax_j):
    return np.maximum(np.minimum(cmin_i, cmin_j), np.minimum(cmax_i, cmax_j) * (1 - np.cos(np.pi * t)**4)**4)

def cos2_diurnal_modulation(t, cmin_i, cmax_i, cmin_j, cmax_j):
    return np.maximum(np.minimum(cmin_i, cmin_j), np.minimum(cmax_i, cmax_j) * (1 - np.cos(np.pi * t)**2)**2)

class DiurnalMeanContactRate:
    def __init__(self, minimum_i=3, maximum_i=22, minimum_j=3, maximum_j=22, diurnal_modulation=cos4_diurnal_modulation):
        self.minimum_i = minimum_i
        self.maximum_i = maximum_i
        self.minimum_j = minimum_j
        self.maximum_j = maximum_j
        self.modulation = diurnal_modulation

    def __call__(self, t):
        return self.modulation(t, self.minimum_i, self.maximum_i, self.minimum_j, self.maximum_j)


