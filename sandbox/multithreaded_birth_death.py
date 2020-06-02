from numba import njit, prange, float64
import numpy as np
from timeit import default_timer as timer

import matplotlib.pyplot as plt

@njit
def diurnal_inception_rate(λmin, λmax, t):
    return np.maximum(λmin, λmax * (1 - np.cos(np.pi * t)**4)**4)

@njit
def simulate_contact(stop_time,
                     previous_stop_time,
                     time,
                     contact_duration, 
                     overshoot,
                     contact,
                     min_inception_rate,
                     max_inception_rate,
                     mean_contact_lifetime):

    if stop_time > time:
        contact_duration = overshoot
    else:
        contact_duration = 0.0 #(stop_time - previous_stop_time) * contact
        overshoot = time - stop_time

    step = 0

    while time < stop_time:
        step += 1

        inception_rate = diurnal_inception_rate(min_inception_rate, max_inception_rate, time)

        if contact: # We have contact. Deactivate!
            time_step = - np.log(np.random.random()) * mean_contact_lifetime

            if time + time_step > stop_time:
                contact_duration += stop_time - time
            else:
                contact_duration += time_step

            time += time_step
            contact = False

        else: # Contact is absent. Proceed to activate.
            time_step = - np.log(np.random.random()) / inception_rate

            time += time_step
            contact = True

    overshoot = (time - stop_time) * contact

    return time, contact, contact_duration, overshoot


@njit(parallel=True)
def simulate_contacts(stop_time,
                      previous_stop_time,
                      time,
                      contact_duration,
                      overshoot,
                      contact,
                      min_inception_rate,
                      max_inception_rate,
                      mean_contact_lifetime):

    for i in prange(len(contact_duration)):

        (time[i],
         contact[i],
         contact_duration[i],
         overshoot[i]) = simulate_contact(stop_time,
                                          previous_stop_time,
                                          time[i],
                                          contact_duration[i],
                                          overshoot[i],
                                          contact[i],
                                          min_inception_rate[i],
                                          max_inception_rate[i],
                                          mean_contact_lifetime[i])
                         

if __name__ == "__main__":
    n = 1000000

    min_inception_rate = 3 * np.ones(n)
    max_inception_rate = 22 * np.ones(n)
    mean_contact_lifetime = 1 / 60 / 24 * np.ones(n)
    time = np.zeros(n)
    contact_duration = np.zeros(n)
    overshoot = np.zeros(n)
    contacts = np.random.choice([False, True], size=n, p=[0.95, 0.05])

    dt = 0.1 * 1 / 24
    steps = int(2 / dt) 
    previous_stop_time = 0.0

    start = timer()

    mean_contact_durations = []
    active_contacts = []
    times = []

    print(np.count_nonzero(contacts))

    active_contacts.append(np.count_nonzero(contacts))
    mean_contact_durations.append(contact_duration.mean())
    times.append(0.0)

    for i in range(steps):

        simulate_contacts((i + 1) * dt,
                          i * dt,
                          time,
                          contact_duration,
                          overshoot,
                          contacts,
                          min_inception_rate,
                          max_inception_rate,
                          mean_contact_lifetime
                         )

        active_contacts.append(np.count_nonzero(contacts))
        mean_contact_durations.append(contact_duration.mean())
        times.append((i+1) * dt)

    end = timer()

    print("Simulation time:", end - start)
    print("Mean contact duration:", contact_duration.mean())

    fig, axs = plt.subplots(nrows=2)

    plt.sca(axs[0])
    plt.plot(times, active_contacts)

    plt.sca(axs[1])
    plt.plot(times[2:-1], mean_contact_durations[2:-1])

    plt.show()
