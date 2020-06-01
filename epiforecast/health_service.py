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
        static_population_network (networkx graph): This is the full (static) network of possible contacts for
                                                    `health_workers` and `community` nodes

                                       
        """
        self.static_population_network=copy.deepcopy(static_population_network)
        self.populace = list(static_population_network.nodes)
        self.population = len(static_population_network)
        
        self.patient_capacity=patient_capacity

        self.recruit_health_workers(static_population_network, health_worker_population)
        
        # Stores a list of Patient classes
        self.patients = []
        # Stores a list of people unable to go into hospital due to maximum capacity 
        self.waiting_list = []

    # We currently `hire` the first `health_worker_population` nodes, in the future we
    # will require a more intelligent hiring method
    def recruit_health_workers(self,static_population_network, health_worker_population):
        
        self.health_workers=list(static_population_network.nodes)[:health_worker_population]

        
    def assign_health_workers(self, patient_address, viable_health_workers, num_assigned=5):

        if num_assigned < len(viable_health_workers) : 
            health_worker_contacts = np.random.choice(viable_health_workers, size=num_assigned, replace=False)
        else:
            health_worker_contacts = viable_health_workers

        health_worker_contacts = [(patient_address, i) for i in health_worker_contacts]    
        return health_worker_contacts
        
    def discharge_and_admit_patients(self, statuses, population_network):
        """
        (public method)
        Contact simulation is always performed on the world network,
        """
        #First modify the current population network
        self.discharge_patients(statuses, population_network)
        self.admit_patients(statuses, population_network)

        print("current patients after hospitalizations", [patient.address for patient in self.patients])
        
    def discharge_patients(self, statuses, population_network):
        '''
        (private method)
        removes a patient from self.patients and reconnect the patient with their neighbours
        if their status is no longer H
        '''
        for i, patient in enumerate(self.patients):
            if statuses[patient.address] != 'H' :
                #discharge_patient 
                population_network.remove_edges_from(patient.health_worker_contacts)
                population_network.add_edges_from(patient.community_contacts)
                print("Discharging patient", patient.address,
                      "with status", statuses[patient.address])
                del self.patients[i]
                
    def admit_patients(self, statuses, population_network):
        '''
        (private method)
        Method to find unoccupied beds, and admit patients from the populace (storing their details)
        '''
        if len(self.patients) < self.patient_capacity:
            # find unoccupied beds
            current_patients = [patient.address for patient in self.patients]
            
            populace =[w for w in filter(lambda w: w not in current_patients, self.populace)]

            hospital_seeking = [i for i in populace if statuses[i] == 'H']
            print("people seeking hospitalization", hospital_seeking)

            hospital_beds=self.patient_capacity - len(self.patients)

            if not isinstance(hospital_seeking, list): # if it's a scalar, not array
                    hospital_seeking=[hospital_seeking]

                    
            if len(hospital_seeking) > hospital_beds:
                patient_admissions = hospital_seeking[:hospital_beds]
                # if we reach capacity and have hospital seekers remaining
                # set hospital seekers back to i
                patient_waiting_list = hospital_seeking[hospital_beds:]
                self.waiting_list.append(patient_waiting_list)

                for patient in patient_waiting_list:
                    statuses[patient] = 'I'
                print("Those placed on a waiting list as unable to get a bed ", hospital_seeking)
            else:
                patient_admissions = hospital_seeking

            #find available health workers
            sick_people = patient_admissions + [ patient.address for patient in self.patients] 
            viable_health_workers = [w for w in filter(lambda w: w not in sick_people, self.health_workers)]

            for patient in patient_admissions:
                # obtain the patients seeking to be hospitalized (state = 'H') 

                
                # create new edge between patient and corresponding health_workers
                health_worker_contacts = self.assign_health_workers(patient,viable_health_workers)

                community_contacts =  list(self.static_population_network.edges(patient))
                # store patient details
                new_patient = Patient(patient,
                                      community_contacts,
                                      health_worker_contacts)
                                                        
                self.patients.append(new_patient)

                # admit patient 
                population_network.remove_edges_from(new_patient.community_contacts)
                population_network.add_edges_from(new_patient.health_worker_contacts)
                print("Admitting patient from", new_patient.address,
                      "with assigned health workers", health_worker_contacts)

                 
            
            
class Patient:
    """
    Container for current patients in hospital
    """
    def __init__(self, address, community_contacts, health_worker_contacts):
        """
        Args
        ----
        address (int): This gives location of the patient in the `static_population_network`
                      (with respect to `static_population_network.node`)

        community_contacts (list of tuples): list of edges to neighbours in `static_population_network`
                                             these are stored here while the patient is in hospital

        health_worker_contacts (list of tuples): a list of edges connecting patient to assigned health workers
        """
        self.address = address
        self.community_contacts = community_contacts
        self.health_worker_contacts = health_worker_contacts
    
            
