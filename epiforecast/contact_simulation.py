import numpy as np
import scipy.sparse as scspa

cos4_diurnal_modulation = lambda t, cmin, cmax: max(cmin, cmax * (1 - np.cos(np.pi * t)**4)**4)
cos2_diurnal_modulation = lambda t, cmin, cmax: max(cmin, cmax * (1 - np.cos(np.pi * t)**2)**2)

class DiurnalContactModulation:
    def __init__(self, peak=22, minimum=3,
                 diurnal_modulation = cos4_diurnal_modulation):

        self.peak_mean_contacts = peak
        self.minimum_mean_contacts = minimum
        self.modulation = diurnal_modulation

    def mean_contact_rate(self, t):
        return self.modulation(t, self.minimum_mean_contacts, self.peak_mean_contacts)





class ContactSimulator:
    """
    Simulator tools for time-dependent contact networks.
    """

    def __init__(self, n_contacts = None, initial_fraction_active_contacts = 0.0,
                 active_contacts = None, start_time = 0.0):
        """
        Args
        ----
            n_contacts (int): number of contacts (default: None)

            active_contacts (np.array): an array of booleans of length `n_contacts` indicating which
                                        contacts are active (default: None)

            initial_fraction_active_contacts (float): the initial fraction of active contacts. Must be between 0 and 1.

            start_time (float): the start time of the simulation in days
        """

        self.time = start_time

        if active_contacts is not None:
            self.active_contacts = active_contacts
        
        else:
            if n_contacts is None:
                raise ValueError("n_contacts must be specified if active_contacts is not specified!")

            active_probability = initial_fraction_active_contacts
            inactive_probability = 1 - active_probability

            self.active_contacts = np.random.choice([False, True],
                                                    size = n_contacts, 
                                                    p = [inactive_probability, active_probability])

        self.n_contacts = len(self.active_contacts)

        # Contact duration within a simulation interval
        self.contact_duration = np.zeros(self.n_contacts)
        self.interval_stop_time = 0.0
        self.interval_start_time = 0.0
        self.interval_steps = 0

        # We use a Gillespie algorithm based on random time steps, which means
        # we overshoot the end of an interval
        self.overshoot_contact_duration = np.zeros(self.n_contacts)
        self.overshoot_time = 0.0
        
    def simulate_contact_duration(self, stop_time,
                                  mean_contact_duration = 45 / 60 / 60 / 24, # 45 seconds, in units of "days"
                                  modulation = DiurnalContactModulation(peak=22, minimum=3)):
        """
        Simulate time-dependent contacts with a birth/death process.
        """

        # Capture the contact duration associated with the 'overshoot',
        # or the difference between the current time of the contact state, and the
        # stop time of the previous interval
        self.contact_duration = self.overshoot_contact_duration

        # Initialize the number of steps
        self.interval_steps = 0

        # Perform the Gillespie simulation
        while self.time < stop_time:
            
            n_active_contacts = np.count_nonzero(self.active_contacts)

            activation_rate   = modulation.mean_contact_rate(self.time) * (self.n_contacts - n_active_contacts)
            deactivation_rate = n_active_contacts / mean_contact_duration

            # Generate exponentially-distributed random time step:
            time_step = -np.log(np.random.random()) / (deactivation_rate + activation_rate)

            if self.time + time_step > stop_time: # the next event occurs after the end of the simulation

                # Add the part of the contact duration occuring within the current interval
                self.contact_duration += (stop_time - self.time) * self.active_contacts

                # Store the "overshoot time" and contact duration during overshoot 
                # for use during a subsequent simulation.
                self.overshoot_time = (self.time + time_step - stop_time)
                self.overshoot_contact_duration = self.overshoot_time * self.active_contacts

            else: # the next event occurs within the simulation interval
                self.contact_duration += time_step * self.active_contacts

            self.time += time_step
            self.interval_steps += 1

            # Because a random, exponentially-distributed amount of time has elapsed, 
            # an "event" occurs (of course)! The event is the activation or deactivation
            # of a contact.
            deactivation_probability = deactivation_rate / (deactivation_rate + activation_rate)
  
            # Draw from uniform random distribution on [0, 1) to decide
            # whether to activate or deactivate contacts
            if np.random.random() < deactivation_probability: # deactivate contacts

                k = np.random.choice(n_active_contacts) # select an active contact
                index_to_move = np.where(self.active_contacts)[0][k] # piece of black magic, woohoo!
                self.active_contacts[index_to_move] = False
  
            else: # activate contacts

                k = np.random.choice(self.n_contacts - n_active_contacts) # select an inactive contact
                index_to_move = np.where(~self.active_contacts)[0][k] 
                self.active_contacts[index_to_move] = True

        # Record the start and stop times of the current simulation interval
        self.interval_start_time = self.interval_stop_time
        self.interval_stop_time = stop_time

    def generate_average_contact_duration(self, time_interval, **kwargs):
        self.simulate_contact_duration(time_interval, **kwargs)

        return self.contact_duration / time_interval
