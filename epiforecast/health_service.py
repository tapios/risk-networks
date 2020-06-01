import copy
import numpy as np
import networkx as nx
class HealthService:
    """
    Class to represent the actions of a health service for the population,
    This class is initiated with a full static network of hospital_beds,
    health_workers and community known as the `world_network`. The important
    structure of this network are the hospital_beds, that are nodes linked
    to the community network, only through a layer of health workers:

    |-----------------------World network------------------------|

    [hospital_bed 1]---[health_worker 1: 'S']---[community 1: 'S'] 
          |         \ /                      \ /    
          |          X                        X-[community 2: 'I'] 
          |         / \                      / \    
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
        population_network=copy.deepcopy(self.world_network)        
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

        self.__set_edge_weights(averaged_contact_duration, population_network)     
        print("current patients after hospitalizations", [patient["address"] for patient in self.patients])
    def __discharge_patients(self, node_statuses, population_network):
        '''
        (private method)
        removes a patient from self.patients and reconnect the patient with their neighbours
        if their status is no longer H
        '''
        for i,patient in enumerate(self.patients):
            if node_statuses[patient["address"]] != 'H' :
                #discharge_patient 
                population_network.remove_edges_from(patient["hospital_contacts"])
                population_network.add_edges_from(patient["community_contacts"])
                print("Discharging patient", patient["address"],
                      "from bed", patient["hospital_bed"], 
                      "with status", node_statuses[patient["address"]])
                del self.patients[i]
                
    def __admit_patients(self, node_statuses, population_network,):
        '''
        (private method)
        Method to find unoccupied beds, and admit patients from the populace (storing their details)
        '''
        if len(self.patients) < self.patient_capacity:
            #find unoccupied beds
            occupied_beds = [self.patients[i]["hospital_bed"] for i in range(len(self.patients))]
            unoccupied_beds=np.delete(np.arange(self.patient_capacity), occupied_beds)
            current_patients = [patient["address"] for patient in self.patients]
            
            populace = np.arange(len(self.world_network.nodes))
            populace = np.delete(populace,current_patients)
            populace = np.delete(populace,np.arange(self.node_identifiers["hospital_beds"].size))
            
            hospital_seeking=[i for i in populace if node_statuses[i] == 'H']
            print("people seeking hospitalization", hospital_seeking)
            for bed in unoccupied_beds:
                #obtain the patients seeking to be hospitalized (state = 'H') 
                if not isinstance(hospital_seeking, list): # if it's a scalar, not array
                    hospital_seeking=[hospital_seeking]
                if (len(hospital_seeking) > 0):                    
                    new_patient = hospital_seeking[0]
                    #create new edge between patient and corresponding health_workers
                    hospital_contacts=list(copy.deepcopy(self.world_network.neighbors(bed)))
                    hospital_contacts=[(i,new_patient) for i in hospital_contacts]
                    #store patient details
                    new_patient_details = {"address"            : new_patient,
                                           "community_contacts" : copy.deepcopy(self.world_network.edges(new_patient)),
                                           "hospital_bed"       : bed,
                                           "hospital_contacts"  : hospital_contacts}
                    self.patients.append(new_patient_details)

                    # admit patient 
                    population_network.remove_edges_from(new_patient_details["community_contacts"])
                    population_network.add_edges_from(new_patient_details["hospital_contacts"])
                    print("Admitting patient from", new_patient_details["address"],
                          "into bed", new_patient_details["hospital_bed"])

                    #remove hospital seeker
                    hospital_seeking.pop(0)
                    
            
            #End cases - a) if we have empty beds remaining
            #            b) if we reach capacity and have hospital seekers remaining
            occupied_beds = [self.patients[i]["hospital_bed"] for i in range(len(self.patients))]
            unoccupied_beds=np.delete(np.arange(self.patient_capacity), occupied_beds)
            if unoccupied_beds.size > 0:
                population_network.remove_nodes_from(unoccupied_beds)
                                       
            # set hospital seekers back to i
            if (len(hospital_seeking) > 0):
                node_statuses[hospital_seeking] = 'I'
                print("those unable to get a bed ", hospital_seeking)

    def __set_edge_weights(self, averaged_contact_duration, population_network):

        # There are 3 steps to convert the averaged_contact_duration on the world_network
        # into the `durations` array on the population network 
        # 1) We first remove values corresponding to edges on the edge list that relate to 
        #    a) Patient contacts in the community/health workers
        #    b) Unoccupied hospital beds
        # 2) We change the values corresponding to edges of occupied hospital beds
        #    and sort them so that into the correct `upper triangular` order so they correpond
        # to the

        # 1a)
        # Find occupied beds
        occupied_beds = [self.patients[i]["hospital_bed"] for i in range(len(self.patients))]
        unoccupied_beds=np.delete(np.arange(self.patient_capacity), occupied_beds)
        edges = copy.deepcopy(self.edges)
        durations = copy.deepcopy(averaged_contact_duration)
        # We now remove patient contacts:
        if len(self.patients)>0:
            patient_community_contacts=[]
            
            for i,patient in enumerate(self.patients):
                # First we find the edges involving the patient
                patient_community_contacts.append(np.where((edges[:,0] == patient["address"]) |
                                                           (edges[:,1] == patient["address"]))
                                                 )
                
            patient_community_contacts= np.unique(np.hstack(patient_community_contacts))
            # We must treat edges and durations consistently...
            edges = np.delete(edges,patient_community_contacts, axis=0)
            durations = np.delete(durations,patient_community_contacts, axis=0)  
           
     
        # 1b)
        # We now remove all empty hospital beds:
        unoccupied_bed_contacts=[]
        for i,bed in enumerate(unoccupied_beds):
            unoccupied_bed_contacts.append(np.where((edges[:,0] == bed) |
                                                    (edges[:,1] == bed))
                                          )

        unoccupied_bed_contacts= np.unique(np.hstack(unoccupied_bed_contacts))
        # We must treat edges and durations consistently...            
        edges = np.delete(edges, unoccupied_bed_contacts,axis=0)
        durations = np.delete(durations, unoccupied_bed_contacts,axis=0)

        # 2)
        # We now adjust the edges for each occupied bed
        if len(self.patients)>0:
            for i,patient in enumerate(self.patients):
            
                # We find the edges involving the hospital bed
                # Note: the edgelist is sorted in upp triangular fashion,
                # if edges[:,1] = bed, then it must belong to another patient with edges[:,0] = bed 
                # therefore this can be ignored
                patient_hospital_contacts = np.where(edges[:,0] == patient["hospital_bed"]) 
                patient_hospital_contacts = np.hstack(patient_hospital_contacts)
                # swap bed for patient in edgelist
                # Note: we uniformly change a sorted list index by a larger constant and swap
                # arguments so that it remains sorted wrt first argument
                # in the second index. E.g for patient 6, in bed zero we do:
                # (0,1),(0,2),(0,3) -> (1,0),(2,0),(3,0) -> (1,60),(2,60),(3,60)
                edges[patient_hospital_contacts, 0] = edges[patient_hospital_contacts, 1] 
                edges[patient_hospital_contacts, 1] = patient["address"]

            # Current check, we should now have the correct list of edges matched to weights,
            # however, the ordering will not match the ordered output of `nx.edges(population_network)`.
            # We sort by first edge argument, as we have verified that the above does not change the ordering

            edge_sort_idx= np.argsort(edges[:,0])
            edges=edges[edge_sort_idx,:]
            durations=durations[edge_sort_idx]
            listpop=np.vstack(list(population_network.edges))
            pop_sort_idx=np.argsort(listpop[:,0])
            pop_unsort_idx=np.argsort(pop_sort_idx)
    
            edges=edges[pop_unsort_idx,:]
            #print(edges)
            durations=durations[pop_unsort_idx] 
            #print(population_network.edges)
           
        #Then set weights on new network
        weights = {tuple(edge): durations[i] 
                   for i, edge in enumerate(nx.edges(population_network))}

        #Recall: edges(population_network) will be an ordered list of 2-tuples
        #ordered so that for any pair 'i': tuple[i][0] < tuple[i][1]
        
        #These are the only attributes that are modified when changing edges.
        #names must match those of `KineticModel.diagram_neigh`
        nx.set_edge_attributes(population_network, values=weights, name='SI->E')
        nx.set_edge_attributes(population_network, values=weights, name='SH->E')
        
