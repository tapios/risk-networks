from numba import njit, prange, float64
import numpy as np
from timeit import default_timer as timer

@njit
def diurnal_inception_rate(λmin, λmax, t):
    return np.maximum(λmin, λmax * (1 - np.cos(np.pi * t)**4)**4)

@njit
def simulate_contact(stop_time, previous_stop_time, time, contact_duration, 
                     overshoot_contact_duration, contact,
                     min_inception_rate,
                     max_inception_rate,
                     mean_contact_lifetime):

    if stop_time > time:
        contact_duration = overshoot_contact_duration
    else:
        contact_duration = (stop_time - previous_stop_time) * contact

    while time < stop_time:

        inception_rate = diurnal_inception_rate(min_inception_rate, max_inception_rate, time)

        if contact: # deactivate
            time_step = - np.log(np.random.random()) * mean_contact_lifetime
            contact_duration += time_step
            contact = False
            time += time_step
        else: 
            time += - np.log(np.random.random()) / inception_rate
            contact = True

    if contact:
        overshoot_contact_duration = time - stop_time
        contact_duration -= (time_step - overshoot_contact_duration)
    else:
        overshoot_contact_duration = 0.0

    return time, contact_duration, overshoot_contact_duration


@njit(parallel=True)
def simulate_contacts(stop_time,
                      previous_stop_time,
                      time,
                      contact_duration,
                      overshoot_contact_duration,
                      contact,
                      min_inception_rate,
                      max_inception_rate,
                      mean_contact_lifetime):

    for i in prange(len(contact_duration)):

        (time[i],
         contact_duration[i],
         overshoot_contact_duration[i]) = simulate_contact(stop_time,
                                                           previous_stop_time,
                                                           time[i],
                                                           contact_duration[i],
                                                           overshoot_contact_duration[i],
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
    overshoot_contact_duration = np.zeros(n)
    contact = np.random.choice([False, True], size=n, p=[0.95, 0.05])

    dt = 1 / 24
    previous_stop_time = 0.0

    start = timer()

    for i in range(24):

        simulate_contacts(
                          (i + 1) * dt,
                          i * dt,
                          time,
                          contact_duration,
                          overshoot_contact_duration,
                          contact,
                          min_inception_rate,
                          max_inception_rate,
                          mean_contact_lifetime
                         )

    end = timer()

    print("Simulation time:", end - start)
    print("Mean contact duration:", contact_duration.mean())
