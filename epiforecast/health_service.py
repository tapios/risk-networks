import copy
import numpy as np
import networkx as nx

class HealthService:
    """
    Class to represent the actions of a health service for the population,
    This class is initiated with a full static network of health_workers
    and community known as the `static_population_network`. Patients are
    added from this population by removing their old connectivity and
    adding new contact edges to some `assigned health workers:

    |----------------network------------------|

    [health_worker 1: 'S']---[community 1: 'S']
                          \ /
                           X-[community 2: 'I']
                          / \
    [health_worker 2: 'S']---[community 3: 'H']

    |-----------------------------------------|

    The primary method of this class is:

    population_network = `discharge_and_obtain_patients(current_statuses)`

    The output is a network identified with only human beings, known as the `population network`
    and the connectivity would be given by the following:

    (obtain) Any `community` or `health_worker` node that (and not already a `patient`)
    with status 'H' (hospitalized)  will lose their edges connecting them to current
    neighbours and gain the edges of some health workers. If there is capacity to do so.
    We refer to them as a `patient`.

    (discharge) Any `patient` with status !='H' (i.e it is now resistant 'R' or deceased 'D') will lose
    the edges to their current `health_worker` neighbours and regain thier original neighbours.

    We perform first (discharge) to empy beds, then (obtain) to gain new patients in the function.

    For example. Applying this function to the world network above yields the following.
    1) discharge: There is noone to discharge from beds 1,2
    2) obtain: `community 3` has status H and so is placed into hospital. They lose their community edges
               and gain the edges of `hospital bed 1`. We denote them `patient 1`.

    |---------------------Population network---------------------|

    [Patient 1: 'H']---[health_worker 1: 'S']---[community 1: 'S']
                    \                        \ /
                     \                        /-[community 2: 'I']
                      \                      /
                       [health_worker 2: 'S']

    |------------------------------------------------------------|

    """

    def __init__(self, patient_capacity, health_worker_population, static_population_network):
        """
        Args
        ----
        patient_capacity (int): Total number of patients that can be hospitalized at a given time.

        health_worker_population (int): Number of healthcare workers.

        static_population_network (networkx graph): The full (static) network of possible contacts for
                                                    `health_workers` and `community` nodes.
        """
        self.static_population_network = copy.deepcopy(static_population_network)
        self.populace = list(static_population_network.nodes)
        self.population = len(static_population_network)
        self.patient_capacity = patient_capacity

        self.recruit_health_workers(static_population_network, health_worker_population)

        # List of Patient classes
        self.patients = []

        # List of people unable to go into hospital due to maximum capacity
        self.waiting_list = []

    def recruit_health_workers(self, static_population_network, health_worker_population):
        """
        Hire the first `health_worker_population` nodes. In the future we
        will require a more intelligent hiring method
        """
        self.health_workers = list(static_population_network.nodes)[:health_worker_population]

    def assign_health_workers(self, patient_address, viable_health_workers, num_assigned=5):
        """
        Assign health workers to a patient.
        """
        if num_assigned < len(viable_health_workers):
            health_worker_contacts = np.random.choice(viable_health_workers, size=num_assigned, replace=False)
        else:
            health_worker_contacts = viable_health_workers

        health_worker_contacts = [ (patient_address, i) for i in health_worker_contacts ]

        return health_worker_contacts

    def discharge_and_admit_patients(self, statuses, population_network):
        """
        Discharge and admit patients.
        """
        # First modify the current population network
        discharged_patients = self.discharge_patients(statuses, population_network)
        admitted_patients = self.admit_patients(statuses, population_network)

        print("Current patients", [patient.address for patient in self.patients])

        return admitted_patients, discharged_patients

    def discharge_patients(self, statuses, population_network):
        """
        Removes a patient from self.patients and reconnect the patient with their neighbours
        if their status is no longer H.
        """

        discharged_patients = []

        for i, patient in enumerate(self.patients):
            if statuses[patient.address] != 'H': # Discharge_patient

                population_network.remove_edges_from(patient.health_worker_contacts)
                population_network.add_edges_from(patient.community_contacts)

                print("Discharging patient", patient.address,
                      "with status", statuses[patient.address])

                discharged_patients.append(patient)

        self.patients = [ p for p in filter(lambda p: p not in discharged_patients, self.patients) ]

        print("Remaining patients after discharge", [p.address for p in self.patients])

        return discharged_patients

    def admit_patients(self, statuses, population_network):
        """
        Method to find unoccupied beds, and admit patients from the community (storing their details).
        """

        admitted_patients = []

        if len(self.patients) < self.patient_capacity:
            # Find unoccupied beds
            current_patients = [patient.address for patient in self.patients]

            populace = [w for w in filter(lambda w: w not in current_patients, self.populace)]
            hospital_seeking = [i for i in populace if statuses[i] == 'H']

            print("Those seeking hospitalization", hospital_seeking)

            hospital_beds = self.patient_capacity - len(self.patients)

            if not isinstance(hospital_seeking, list): # if it's a scalar, not array
                hospital_seeking = [hospital_seeking]

            if len(hospital_seeking) > hospital_beds:
                patient_admissions = hospital_seeking[:hospital_beds]

                # If we reach capacity and have hospital seekers remaining
                # set hospital seekers back to i
                patient_waiting_list = hospital_seeking[hospital_beds:]
                self.waiting_list.extend(patient_waiting_list)

                statuses.update({node: 'I' for node in patient_waiting_list})
                print("Those placed on a waiting list as unable to get a bed ", patient_waiting_list)

            else:
                patient_admissions = hospital_seeking

            # Find available health workers
            sick_people = patient_admissions + [ patient.address for patient in self.patients]
            viable_health_workers = [w for w in filter(lambda w: w not in sick_people, self.health_workers)]

            for patient in patient_admissions:
                # Create new edge between patient and corresponding health_workers
                health_worker_contacts = self.assign_health_workers(patient, viable_health_workers)

                community_contacts = list(self.static_population_network.edges(patient))

                # Store patient details
                new_patient = Patient(patient,
                                      community_contacts,
                                      health_worker_contacts)

                # Obtain the patients seeking to be hospitalized (state = 'H').
                self.patients.append(new_patient)
                admitted_patients.append(new_patient)

                # Admit patient
                population_network.remove_edges_from(new_patient.community_contacts)
                population_network.add_edges_from(new_patient.health_worker_contacts)

                print("Admitting patient from", new_patient.address,
                      "with assigned health workers", health_worker_contacts,
                      "and severing community contacts", new_patient.community_contacts)

        return admitted_patients



class Patient:
    """
    Patient in hospital and their information.
    """
    def __init__(self, address, community_contacts, health_worker_contacts):
        """
        Args
        ----
        address (int): Location of the patient in the `static_population_network`.

        community_contacts (list of tuples): List of edges in the `static_population_network`.
                                             These are stored here while the patient is in hospital

        health_worker_contacts (list of tuples): a list of edges connecting patient to assigned health workers
        """
        self.address = address
        self.community_contacts = community_contacts
        self.health_worker_contacts = health_worker_contacts
