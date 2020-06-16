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
    FractionalUserBase takes a random subset of the population and contructs a subgraph of the largest
    component within this fraction. 
    """
    def __init__(self,
                 full_contact_network,
                 user_fraction):
        
        user_base=[]
        scale_factor=1.0
        while len(user_base)< 0.9*user_fraction*len(full_contact_network):
            nodes_size= min(int(scale_factor*user_fraction*len(full_contact_network.nodes)),len(full_contact_network.nodes))
            users = np.random.choice(list(full_contact_network.nodes),nodes_size, replace=False) 
            user_base_fractured=full_contact_network.subgraph(users)
            user_base=max(nx.connected_components(user_base_fractured),key=len)
            scale_factor*=1.1
            
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


                    
def contiguous_indicators(graph,
                          subgraph):
    """
    A function that returns user base subgraph indicators and corresponding edge/node
    lists with attributes "exterior" and "interior".
    """
    edge_indicator_dict = {edge: "interior" if edge in subgraph.edges() else "exterior" for edge in graph.edges()}    
    node_indicator_dict = {node: "interior" if node in subgraph.nodes() else "exterior" for node in graph.nodes()}    

    edge_indicator_list = []
    for key, value in  zip(edge_indicator_dict.keys(), edge_indicator_dict.values()):
        edge_indicator_list.append([key[0], key[1], value])
    
    node_indicator_list = []
    for key, value in  zip(node_indicator_dict.keys(), node_indicator_dict.values()):
        node_indicator_list.append([key, value])
        
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
                    
    return interior_nodes, boundary_nodes, mean_neighbors_exterior, edge_indicator_list, node_indicator_list



def assign_user_connectivity_to_contact_network(full_contact_network,
                                                user_base,
                                                homogeneous_score = False):
    """
    A function to assign a score in [0,1] to the nodes of the full contact network that are in the user base.
    The score gives an measure of the connectivity of the user with the user base versus the connectivity
    with the rest of the network. (0 for interior node, near 1 for 'boundary node')
    (NB the `user_base` is a read-only type object so we must apply this to the full network)

    Args
    ----
    full_contact_network (networkx Graph): the full contact network
    user_base       (networkx Graph view): A subgraph of full_contact_network, the nodes are `users`
    """
    if homogeneous_score is True:
        connectivity_score = [ 1.0 for node in user_base.nodes ]
    else:
        connectivity_score = [(len(full_contact_network.edges(node)) - len(user_base.edges(node))) /
                              len(full_contact_network.edges(node))
                              for node in user_base.nodes]
    
    user_scores = { node: connectivity_score[i] for i, node in enumerate(user_base.nodes) }
    nx.set_node_attributes(full_contact_network, values = user_scores, name = "user_connectivity")
