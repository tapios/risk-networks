import copy
import numpy as np

class HealthService:
    """
    Class to represent the actions of a health service for the population,
    This class is initiated with a full static network of hospital_beds,
    health_workers and community known as the `world_network`. The important
    structure of this network are the hospital_beds, that are nodes linked
    to the community network, only through a layer of health workers:

    |-----------------------World network------------------------|

    [hospital_bed 1]---[health_worker 1: 'S']---[community 1: 'S'] 
                    \ /                      \ /    
                     X                        X-[community 2: 'I'] 
                    / \                      / \    
    [hospital_bed 2]---[health_worker 2: 'S']---[community 3: 'H']

    |------------------------------------------------------------|

    The primary method of this class is:

    population_network = `discharge_and_obtain_patients(current_statuses)`

    The output is a network identified with only human beings, known as the `population network`
    (no hospital beds) and the connectivity is different from world network as given by the
    following:

    (obtain) Any `community` or `health_worker` node that (and not already a `patient`)
    with status 'H' (hospitalized)  will lose their edges connecting them to current
    neighbours and gain the edges of an unoccupied hospital. We refer to them as a `patient`.

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

    def __init__(self, node_identifiers, world_network, edges):
        """
        Args
        ----
        world_network (networkx graph): This is the full (static) network of possible contacts for
                                        `hospital_beds`, `health_workers` and `community` nodes

        node_identifiers (dict [np.array]): a dict of size 3;
                          ["hospital_beds"] contains the node indices of the hospital beds
                          ["health_workers"] contains the node indices of the health workers
                          ["community"] contains the node indices of the community
                                       
        """
        self.world_network=world_network
        self.edges=edges
        self.node_identifiers = node_identifiers
        self.patient_capacity = node_identifiers["hospital_beds"].size   
        # Stores { "hospital bed" , "address" , "population_contacts"} of patient nodes in a dict 
        self.patients = []

        
    def create_population_network(self):
        """
        Creates the network for KineticModel to simulate on, includes health_workers and
        community nodes, but without hospital beds
        """
        population_network=copy.deepcopy(world_network)        
        population_network.remove_nodes_from(self.node_identifiers["hospital_beds"])
        return population_network
        
    def discharge_and_admit_patients(self, population_statuses, averaged_contact_duration, population_network):
        """
        (public method)
        Contact simulation is always performed on the world network, therefore 
        """
        #First create the current contact network
        self.__discharge_patients(population_statuses, population_network)
        self.__admit_patients(population_statuses, population_network)

        self.__set_edge_weights(averaged_contact_duration, population, population_network)     
        
    def __discharge_patients(self, node_statuses, population_network):
        '''
        (private method)
        removes a patient from self.patients and reconnect the patient with their neighbours
        if their status is no longer H
        '''
        for i,patient in enumerate(self.patients):
            if node_statuses[patient["address"]] != 'H' :
                #discharge_patient 
                nx.remove_edges_from(patient["hospital_contacts"])
                nx.add_edges_from(patient["community_contacts"])
                print("Discharging patient ", patient["address"],
                      " from bed ", patient["bed"], 
                      " in state ", node_statuses[patient["address"]])
                del self.patients[i]
                
    def __admit_patients(self, population_network,node_statuses):
        '''
        (private method)
        Method to find unoccupied beds, and admit patients from the populace (storing their details)
        '''
        if len(self.patients) < self.patient_capacity:
            #find unoccupied beds
            occupied_beds = [self.patients[i]["hospital_bed"] for i in range(len(self.patients))]
            unoccupied_beds=np.delete(np.arange(self.patient_capacity), occupied_beds)

            for bed in unoccipied_beds:
                #obtain the patients seeking to be hospitalized (state = 'H') 
                populace = np.hstack([self.node_identifiers["health_workers"] , self.node_identifiers["community"]])
                hospital_seeking=[i for i in populace if node_statuses[i] == 'H']
                if (len(hospital_seeking) > 0):
                    new_patient = hospital_seeking[0]
                    #create new edge for patient to corresponding hospital_workers
                    hospital_contacts=list(copy.deepcopy(self.world_network.neighbors(bed)))
                    hospital_contacts=[tuple(new_patient,i) for i in bed_contacts]
                    #store patient details
                    new_patient_details = {"address"            : new_patient,
                                           "community_contacts" : copy.deepcopy(self.world_network.edges(new_patient)),
                                           "hospital_bed"       : bed,
                                           "hospital_contacts"  : hospital_contacts}
                    self.patients.append(new_patient_details)

                 
                    # admit patient 
                    nx.remove_edges_from(new_patient_details["community_contacts"])
                    nx.add_edges_from(new_patient_details["hospital_contacts"])
                    print("Admitting patient from", new_patient_details["address"],
                          " into bed ", patient["bed"])

                 
                    
    def __set_edge_weights(averaged_contact_duration, population, population_network)     :

        #currently have weights for world_network.
        #need to reorder/remove so the fit population network

        
        #remove patient["community_contact"] weights
        #swap occupied beds for patients (patient["bed"] , worker) = XXX -> (patient["address"], worker) = XXX
        #extend weights by the new edges to patients. 
        #remove weights for hospital bed nodes

        #find occupied beds
        #occupied_beds = [self.patients[i]["hospital_bed"] for i in range(len(self.patients))]
        #unoccupied_beds=np.delete(np.arange(self.patient_capacity), occupied_beds)

        #find edges without the patients
        nonpatient_contacts=np.zeros(len(self.patients),self.edges.shape[0])
        patient_community_contacts=np.zeros(len(self.patients),self.edges.shape[0])
        patient_hospital_contacts=np.zeros(len(self.patients),self.edges.shape[0])
        for i,patient in self.patients:
            #if the patient is part of an edge, remove the corresponding element of the contact duration
                  nonpatient_contacts[i,:] = np.where((self.edges[:,0] != patient["address"]) &
                                                      (self.edges[:,1] != patient["address"]))]

                  patient_community_contacts[i,:] = np.where((self.edges[:,0] == patient["address"]) |
                                                                (self.edges[:,1] == patient["address"]))]
                
                  patient_hospital_contacts[i,:] = np.where((self.edges[:,0] == patient["bed"]) |
                                                            (self.edges[:,1] == patient["bed"]))]

        #Booleans are currently for each patient, use any or all for statement about all the patients
        nonpatient_contacts= np.all(nonpatient_contact,axis=0))
        patient_community_contacts= np.any(nonpatient_contact,axis=0))
        patient_hospital_contacts= np.any(nonpatient_contact,axis=0))
        
        #Work In Progress...
        
        #Then set weights on new network
        weights = {tuple(edge): new_averaged_contact_duration[i] 
                   for i, edge in enumerate(nx.edges(self.population_network))}
        
        #These are the only attributes that are modified when changing edges.
        nx.set_edge_attributes(population_network, values=weights, name='SI->E')
        nx.set_edge_attributes(population_network, values=weights, name='SH->E')
        
