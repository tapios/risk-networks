import numpy as np
import networkx as nx

from timeit import default_timer as timer

from .contact_simulator import ContactSimulator
from .kinetic_model_simulator import KineticModel
from .utilities import not_involving

day = 1
hour = day / 24
minute = hour / 60
second = minute / 60

class EpidemicSimulator:
    """
    Simulates epidemics.
    """
    def __init__(self,
                 contact_network,
                 transition_rates,
                 community_transmission_rate,
                 hospital_transmission_reduction,
                 contact_simulator = True,
                 mean_contact_lifetime = None,
                 static_contact_interval = None,
                 day_inception_rate = None,
                 night_inception_rate = None,
                 health_service = None,
                 start_time = 0.0):
        """
        Build a tool that simulates epidemics.

        Args
        ----

        contact_network (nx.Graph): Network of community members and edges that represent
                                    possible contact between community members.

        transition_rates: Container holding transition rates between states.

        community_transmission_rate (float): Rate of transmission of infection during interaction
                                             between people in the community.

        hospital_transmission_reduction (float): Fractional reduction of rate of transmission of
                                                 infection in hospitals relative to community.

        static_contact_interval (float): Interval over which contact between people is assumed 'static'.
                                         Rapidly fluctuating contact times are averaged over this interval
                                         and then used in kinetic_model.simulate.

        mean_contact_lifetime (float): The *mean* lifetime of a contact between people. Typical values
                                       are O(minutes).

        day_inception_rate (float): The rate of inception of new contacts between people at noon.

        night_inception_rate (float): The rate of inception of new contacts between people at midnight.

        health_service: Manages rewiring of contact_network during hospitalization.

        start_time (float): The initial time of the simulation.

        """

        self.contact_network = contact_network
        self.health_service = health_service

        if health_service is None: # number of contacts cannot change; no buffer needed
            buffer_margin = 1
        else:
            buffer_margin = 1.2 # 20% margin seems conservative

        if contact_simulator:
            self.contact_simulator = ContactSimulator(contact_network,
                                                      day_inception_rate = day_inception_rate,
                                                      night_inception_rate = night_inception_rate,
                                                      mean_event_lifetime = mean_contact_lifetime,
                                                      buffer_margin = buffer_margin,
                                                      start_time = start_time)
        else:
            self.contact_simulator = None

        self.kinetic_model = KineticModel(contact_network = contact_network,
                                          transition_rates = transition_rates,
                                          community_transmission_rate = community_transmission_rate,
                                          hospital_transmission_reduction = hospital_transmission_reduction,
                                          start_time = start_time)

        self.static_contact_interval = static_contact_interval
        self.time = start_time

    def run_with_static_contacts(self, stop_time):
        """
        Run the kinetic model forward until `stop_time` without running the ContactSimulator.
        """

        time_interval = stop_time - self.time

        edges_to_add, edges_to_remove, hospital_admin_wall_time = self.discharge_and_admit_patients()

        start_kinetic_simulation = timer()

        self.kinetic_model.simulate(time_interval)
        self.time = stop_time

        self.report_statuses()
        end_kinetic_simulation = timer()

        print("[ Wall time ]       Kinetic simulation: {:.4f} s".format(end_kinetic_simulation - start_kinetic_simulation))

    def run(self, stop_time):
        """
        Run forward until `stop_time`. Takes a single step when
        `stop_time = self.time + self.static_contact_interval`.
        """

        run_time = stop_time - self.time

        if self.static_contact_interval is None:
            # Take one step using averaged contacts over the whole run_time
            constant_steps = 1
        else:
            # Take constant steps of length self.static_contact_interval, followed by a single ragged step to 
            # update to specified stop_time.
            constant_steps = int(np.floor(run_time / self.static_contact_interval))

        interval_stop_times = self.time + self.static_contact_interval * np.arange(start = 1, stop = 1 + constant_steps)

        # Step forward
        for i in range(constant_steps):

            interval_stop_time = interval_stop_times[i]

            print("")
            print("")
            print("                               *** Day: {:.3f}".format(interval_stop_time))
            print("")

            #
            # Administer hospitalization
            #

            edges_to_add, edges_to_remove, hospital_admin_wall_time = self.discharge_and_admit_patients()

            #
            # Simulate contacts
            #

            contact_simulation_wall_time = self.run_contact_simulator(interval_stop_time, edges_to_add, edges_to_remove)
            
            #
            # Run the kinetic simulation
            #

            start_kinetic_simulation = timer()

            self.kinetic_model.simulate(self.static_contact_interval)
            self.time += self.static_contact_interval

            end_kinetic_simulation = timer()

            #
            # Print a status report with wall times
            #

            self.report_statuses()

            print("[ Wall times ]      Kinetic simulation: {:.4f} s".format(end_kinetic_simulation - start_kinetic_simulation))
            print("                    Contact simulation: {:.4f} s,".format(contact_simulation_wall_time))

            if self.health_service is not None:
                print("                  Hosp. administration: {:.4f} s,".format(hospital_admin_wall_time))

            print("")

        if self.time < stop_time: # take a final ragged stop to catch up with stop_time

            edges_to_add, edges_to_remove = self.discharge_and_admit_patients()

            self.run_contact_simulator(stop_time, edges_to_add, edges_to_remove)

            self.kinetic_model.simulate(self.static_contact_interval)
            self.time = stop_time


    def run_contact_simulator(self, interval_stop_time, edges_to_add, edges_to_remove):        
        """
        Run the contact simulator if dynamic_contacts and set the edge weights of self.contact_network.
        """
        if self.contact_simulator is None:
            raise ValueError("Must use contact_simulator = True when constructing EpidemicSimulator to run contact_simulator.")

        start_contact_simulation = timer()

        self.contact_simulator.run_and_set_edge_weights(stop_time = interval_stop_time,
                                                        edges_to_add = edges_to_add,
                                                        edges_to_remove = edges_to_remove)

        end_contact_simulation = timer()

        return end_contact_simulation - start_contact_simulation

    def discharge_and_admit_patients(self):

        start_health_service_action = timer()

        if self.health_service is not None:
            discharged, admitted = self.health_service.discharge_and_admit_patients(self.kinetic_model.current_statuses,
                                                                                    self.contact_network)

            # Compile edges to add and remove from contact simulation...
            edges_to_remove = set()
            edges_to_add = set()

            current_patients = self.health_service.current_patient_addresses()

            previous_patients = current_patients - {p.address for p in admitted}
            previous_patients.update(p.address for p in discharged)

            # ... ensuring that edges are not removed from previous patients (whose edges were *already* removed),
            # and ensuring that edges are not added to existing patients:
            if len(admitted) > 0:
                for patient in admitted:
                    edges_to_remove.update(filter(not_involving(previous_patients), patient.community_contacts))
                    edges_to_add.update(patient.health_worker_contacts)

            if len(discharged) > 0:
                for patient in discharged:
                    edges_to_remove.update(patient.health_worker_contacts)
                    edges_to_add.update(filter(not_involving(current_patients), patient.community_contacts))

        else:
            edges_to_add, edges_to_remove = set(), set()

        end_health_service_action = timer()

        return edges_to_add, edges_to_remove, end_health_service_action - start_health_service_action

    def report_statuses(self):

        self.n_contacts = nx.number_of_edges(self.contact_network)

        print("")
        print("[ Status report ]          Susceptible: {:d}".format(self.kinetic_model.statuses['S'][-1]))
        print("                               Exposed: {:d}".format(self.kinetic_model.statuses['E'][-1]))
        print("                              Infected: {:d}".format(self.kinetic_model.statuses['I'][-1]))
        print("                          Hospitalized: {:d}".format(self.kinetic_model.statuses['H'][-1]))
        print("                             Resistant: {:d}".format(self.kinetic_model.statuses['R'][-1]))
        print("                              Deceased: {:d}".format(self.kinetic_model.statuses['D'][-1]))

        if self.contact_simulator is not None:
            print("             Current possible contacts: {:d}".format(self.n_contacts))
            print("               Current active contacts: {:d}".format(np.count_nonzero(~self.contact_simulator.active_contacts[:self.n_contacts])))

        print("")


    def set_statuses(self, statuses, time=None):
        """
        Set the statuses of the kinetic_model.
        """
        if time is not None:
            self.time = time

        self.kinetic_model.set_statuses(statuses, time=time)

