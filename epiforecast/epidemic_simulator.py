import numpy as np
import networkx as nx

from timeit import default_timer as timer

from .contact_simulator import ContactSimulator
from .kinetic_model_simulator import KineticModel, print_statuses

day = 1
hour = day / 24
minute = hour / 60
second = minute / 60

class EpidemicSimulator:
    def __init__(self,
                 contact_network,
                 transition_rates,
                 community_transmission_rate,
                 hospital_transmission_reduction,
                 static_contact_interval,
                 mean_contact_lifetime,
                 day_inception_rate = None,
                 night_inception_rate = None,
                 health_service = None,
                 start_time = 0.0):

        self.contact_network = contact_network
        self.health_service = health_service

        contacts_buffer = 0

        if health_service is not None:
            contacts_buffer = 5 * health_service.patient_capacity

        self.contact_simulator = ContactSimulator(contact_network,
                                                    day_inception_rate = day_inception_rate,
                                                  night_inception_rate = night_inception_rate,
                                                   mean_event_lifetime = mean_contact_lifetime,
                                                       contacts_buffer = contacts_buffer,
                                                            start_time = start_time)

        self.kinetic_model = KineticModel(contact_network = contact_network,
                                          transition_rates = transition_rates,
                                          community_transmission_rate = community_transmission_rate,
                                          hospital_transmission_reduction = hospital_transmission_reduction)

        self.static_contact_interval = static_contact_interval
        self.time = start_time

    def run(self, stop_time):

        start_run = timer()

        # Duration of the run
        run_time = stop_time - self.time

        # Number of steps
        constant_steps = int(np.floor(run_time / self.static_contact_interval))

        # Interval stop times
        interval_stop_times = self.time + self.static_contact_interval * np.arange(start = 1, stop = 1 + constant_steps)

        # Step forward
        for i in range(constant_steps):

            interval_stop_time = interval_stop_times[i]

            admitted_patients, living_discharged_patients = [], []

            if self.health_service is not None:
                admitted_patients, living_discharged_patients = (
                    self.health_service.discharge_and_admit_patients(self.kinetic_model.current_statuses,
                                                                     self.contact_network))

            start = timer()

            self.contact_simulator.run_and_set_edge_weights(stop_time=interval_stop_time,
                                                            admitted_patients=admitted_patients,
                                                            discharged_patients=living_discharged_patients)

            self.kinetic_model.simulate(self.static_contact_interval)
            self.time += self.static_contact_interval

            end = timer()

            print("Epidemic day: {: 7.3f}, wall_time: {: 6.3f} s,".format(self.kinetic_model.times[-1], end - start),
                  "mean(w_ji): {: 3.0f} min,".format(self.contact_simulator.contact_duration.mean() / minute),
                  "statuses: ",
                  "S {: 4d} |".format(self.kinetic_model.statuses['S'][-1]),
                  "E {: 4d} |".format(self.kinetic_model.statuses['E'][-1]),
                  "I {: 4d} |".format(self.kinetic_model.statuses['I'][-1]),
                  "H {: 4d} |".format(self.kinetic_model.statuses['H'][-1]),
                  "R {: 4d} |".format(self.kinetic_model.statuses['R'][-1]),
                  "D {: 4d} |".format(self.kinetic_model.statuses['D'][-1]))

            self.contact_simulator.set_time(self.time)

        if self.time != stop_time: # One final step...

            contact_duration = self.contact_simulator.mean_contact_duration(stop_time=stop_time)
            self.kinetic_model.set_mean_contact_duration(contact_duration)
            self.kinetic_model.simulate(stop_time - self.time)
            self.time = stop_time

        end_run = timer()

        print("\n(Epidemic simulation took {:.3f} seconds.)\n".format(end_run-start_run))

    def set_statuses(self, statuses):
        self.kinetic_model.set_statuses(statuses)
