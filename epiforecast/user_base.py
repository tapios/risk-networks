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
    ContiguousUserBase takes a given sized population of the user base and tries to form a complete graph
    about a seed user of this size. Note, this procedure will lead to (slightly) more than the fraction specified
    """
    def __init__(self,
                 full_contact_network,
                 user_fraction,
                 seed_user=None):

        if seed_user is None:
            seed_user = np.random.choice(list(full_contact_network.nodes), replace=False) 

        #get cliques about users (maximal complete graphs)
        node_clique=list(nx.find_cliques(full_contact_network))

        user_population = int(user_fraction * len(full_contact_network.nodes))
        users = [seed_user]
        idx=0
        while len(users) < user_population and idx<len(users):
            #get maximal clique about a new user
            new_users = node_clique[users[idx]]
            new_users = [user for user in filter(lambda u: u not in users, new_users)]
            users.extend(new_users)
            idx=idx+1

        self.contact_network = full_contact_network.subgraph(users)
