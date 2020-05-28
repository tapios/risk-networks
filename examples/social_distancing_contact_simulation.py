import os, sys; sys.path.append(os.path.join(".."))

import numpy as np
import matplotlib.pyplot as plt

from epiforecast.contact_simulation import ContactSimulator, DiurnalContactModulation

np.random.seed(1234)

time_step = 1/24

def simulate_one_day(simulator, peak_mean_contacts, minimum_mean_contacts=5):
    """
    Return the mean contact duration and active contacts at every hour
    over a day of contact simulation by `simulator`.
    """

    diurnal_modulation = DiurnalContactModulation(peak = peak_mean_contacts,
                                                  minimum = minimum_mean_contacts)

    start_time = simulator.interval_stop_time # stop time for the previous interval

    active_contacts = []
    mean_contact_duration = []

    for i in range(int(1/time_step)):
        simulator.simulate_contact_duration(stop_time = start_time + (i + 1) * time_step,
                                            modulation = diurnal_modulation) 

        mean_interval_contact_duration = simulator.contact_duration.mean()
        mean_contact_duration.append(mean_interval_contact_duration)
        active_contacts.append(np.count_nonzero(simulator.active_contacts))
    
    return mean_contact_duration, active_contacts

#####
##### Generate a few time series
#####

n_contacts = 10000

as_usual_simulator   = ContactSimulator(n_contacts = n_contacts,
                                        initial_fraction_active_contacts = 0.1)

distancing_simulator = ContactSimulator(active_contacts=as_usual_simulator.active_contacts)

as_usual_contacts = []
as_usual_durations = []
distancing_contacts = []
distancing_durations = []

days = 4
times = time_step * np.arange(24 * days)

for day in range(days):
    
    as_usual_mean_contacts = 30
    distancing_mean_contacts = 30 - 7 * day

    durations, contacts = simulate_one_day(as_usual_simulator, 
                                           peak_mean_contacts = as_usual_mean_contacts)

    as_usual_contacts = as_usual_contacts + contacts
    as_usual_durations = as_usual_durations + durations

    durations, contacts = simulate_one_day(distancing_simulator,
                                           peak_mean_contacts = distancing_mean_contacts)

    distancing_contacts = distancing_contacts + contacts
    distancing_durations = distancing_durations + durations


# Convert data to arrays and plot
#as_usual_contacts = np.array(as_usual_contacts)
#as_usual_durations = np.array(as_usual_durations)
#distancing_contacts = np.array(distancing_contacts)
#distancing_durations = np.array(distancing_durations)

fig, axs = plt.subplots(nrows=2, figsize=(9, 6))
minute = 1 / 60 / 24

plt.sca(axs[0])
plt.plot(times, as_usual_contacts, "-", label="No distancing")
plt.plot(times, distancing_contacts, "-", label="Social distancing")

plt.sca(axs[1])
plt.plot(times, as_usual_durations / minute, "-", label="No distancing")
plt.plot(times, distancing_durations / minute, "-", label="Social distancing")

# Format plot
plt.sca(axs[0])
plt.ylabel("Number of active contacts")
plt.legend()

plt.sca(axs[1])
plt.xlabel("Time (days)")
plt.ylabel("Mean contact duration (minutes)")

plt.savefig("contacts_with_social_distancing.png", dpi=480)
plt.show()
