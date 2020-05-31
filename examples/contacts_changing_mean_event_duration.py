import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

from epiforecast.fast_contact_simulator import ContactSimulator, DiurnalMeanContactRate

np.random.seed(1234)

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(9, 6))

def plot_contact_simulation(n_contacts,
                            mean_event_duration = 1 / 60 / 24, 
                            mean_contact_rate = DiurnalMeanContactRate(),
                            time_step = 0.01 * 1/24
                            ):

    simulator = ContactSimulator(n_contacts = n_contacts, 
                                 mean_event_duration = mean_event_duration,
                                 mean_contact_rate = mean_contact_rate,
                                 initial_fraction_active_contacts = 0.1)

    mean_contact_duration = []
    active_contacts = []
    times = []
    
    # Generate a time-series of contact durations and average number of active contacts
    days = 2

    for i in range(int(days / time_step)):
        stop_time = (i + 1) * time_step
        simulator.run(stop_time = stop_time)
        mean_contact_duration.append(simulator.interval_contact_duration.mean())
        times.append(i * time_step)

        active_contacts.append(np.count_nonzero(simulator.active_contacts))
    
    # Store results
    mean_contact_duration = np.array(mean_contact_duration)
    times = np.array(times)
    active_contacts = np.array(active_contacts)

    # Plot
    second = 1 / 60 / 60 / 24
    minute = 60 * second

    label = "$ \mu^{{-1}} = {:.1f} $ min".format(mean_event_duration / minute)

    plt.sca(axs[0])
    plt.plot(times, active_contacts, "-", label=label)

    plt.sca(axs[1])
    plt.plot(times, mean_contact_duration / second, "-", alpha=0.5, label=label)

    return times, mean_contact_duration, active_contacts

# Loop over a bunch of network sizes
minute = 1 / 60 / 24

for mean_event_duration in (
                             1 * minute,
                             10 * minute,
                             60 * minute,
                            ):

    n_contacts = 5000
    mean_contact_rate = DiurnalMeanContactRate()
    start = timer()
    plot_contact_simulation(n_contacts = n_contacts, 
                            mean_event_duration = mean_event_duration,
                            mean_contact_rate = mean_contact_rate
                           )
    end = timer()

    print("Simulated", n_contacts,
          "contacts in {:.3f} seconds".format(end - start))

# Format a plot prettily
plt.sca(axs[0])
plt.ylabel("Number of active contacts")
plt.legend()

plt.sca(axs[1])
plt.xlabel("Time (days)")
plt.ylabel("Mean contact duration (seconds)")

plt.savefig("contact_simulation.png", dpi=480)
plt.show()
