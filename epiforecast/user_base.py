import networkx as nx
import numpy as np

class FullUserBase:
    """
    A class to store which subset of the population are being modeled by the Master Equations
    FullUserBase is just the `full_contact_network`.
    """
    def __init__(self,
                 full_contact_network):

        self.contact_network=full_contact_network


        
    
class FractionalUserBase:
    """
    A class to store which subset of the population are being modeled by the Master Equations
    FractionalUserBase takes a random subset of the population and contructs a subgraph without
    isolated nodes. Note, this procedure will lead to (slightly) less than the fraction specified
    """
    def __init__(self,
                 full_contact_network,
                 user_fraction):

        users = np.random.choice(list(full_contact_network.nodes),int(user_fraction*len(full_contact_network.nodes)), replace=False) 

        user_base_with_isolates=full_contact_network.subgraph(users)
        isolates = nx.isolates(user_base_with_isolates)
        
        #removes isolated nodes from user list, create new subgraph (as isolated nodes can't get sick)
        users = [w for w in filter(lambda w: w not in isolates, users)]
        user_base = full_contact_network.subgraph(users)

        self.contact_network = full_contact_network.subgraph(user_base)
        

class ContiguousUserBase:
    """
    A class to store which subset of the population are being modeled by the Master Equations
    ContiguousUserBase takes a given sized population of the user base and tries to form an island of users
    around a seed user by iteration, in each iteration we add nodes to our subgraph by either
    1) "neighbor" - [recommended] adds the neighborhood about the seed user and moving to a new user as a seed
    2) "clique" - adds the maximal clique about the seed user and  moving to a new user as a seed 
    """
    def __init__(self,
                 full_contact_network,
                 user_fraction,
                 method="neighbor",
                 seed_user=None):

        if seed_user is None:
            seed_user = np.random.choice(list(full_contact_network.nodes), replace=False) 

        user_population = int(user_fraction * len(full_contact_network.nodes))
        users = [seed_user]
        idx=0
        # method to build graph based on neighbours [recommended]
        if method == "neighbor":
            while len(users) < user_population and idx<len(users): 
                new_users = full_contact_network.neighbors(users[idx])
                new_users = [user for user in filter(lambda u: u not in users, new_users)]
                users.extend(new_users)
                idx=idx+1
                
        # method to build graph based on cliques
        elif method == "clique": # can be very slow.
            #get cliques about users (maximal complete graphs)
            node_clique=list(nx.find_cliques(full_contact_network))
            while len(users) < user_population and idx<len(users):
                new_users = node_clique[users[idx]]
                new_users = [user for user in filter(lambda u: u not in users, new_users)]
                users.extend(new_users)
                idx=idx+1
        else:
            raise ValueError("unknown method, choose from: neighbor, clique")    
        self.contact_network = full_contact_network.subgraph(users)


                    
def contiguous_indicators(graph, subgraph):

    interior_nodes=0
    boundary_nodes=0
    exterior_neighbor_count=0
    for node in subgraph.nodes:

        if list(subgraph.neighbors(node)) == list(graph.neighbors(node)):
            interior_nodes+=1
        else:
            boundary_nodes+=1
            #count number of exterior neighbors of an boundary node
            exterior_neighbors=[nbr for nbr in filter(lambda nbr: nbr not in list(subgraph.neighbors(node)),list(graph.neighbors(node)))]
            exterior_neighbor_count+=len(exterior_neighbors)

    mean_neighbors_exterior=exterior_neighbor_count / boundary_nodes
                    
    return interior_nodes, boundary_nodes, mean_neighbors_exterior