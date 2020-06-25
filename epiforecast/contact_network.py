import numpy as np
import scipy.sparse as scspa
import networkx as nx

class ContactNetwork:
    """
    Store and mutate a contact network
    """

    health_workers_id = 'HCW'
    community_id      = 'CITY'
    health_workers_index = 0
    community_index      = 1

    # TODO extract into Glossary class
    age_group = 'age_group'

    lambda_min = 'minimum_contact_rate'
    lambda_max = 'maximum_contact_rate'

    WJI = 'edge_weights'

    e_to_i = 'exposed_to_infected'
    i_to_h = 'infected_to_hospitalized'
    i_to_r = 'infected_to_resistant'
    i_to_d = 'infected_to_deceased'
    h_to_r = 'hospitalized_to_resistant'
    h_to_d = 'hospitalized_to_deceased'

    @classmethod
    def from_networkx_graph(
            cls,
            graph):
        """
        Create an object from a nx.Graph object

        Input:
            graph (nx.Graph): an object to use as a contact network graph

        Output:
            contact_network (ContactNetwork): initialized object
        """
        edges       = np.array(graph.edges)
        node_groups = {
                ContactNetwork.health_workers_index : np.array([]),
                ContactNetwork.community_index      : np.array(graph.nodes) }

        return cls(edges, node_groups)

    @classmethod
    def from_files(
            cls,
            edges_filename,
            identifiers_filename):
        """
        Create an object from files that contain edges and identifiers

        Input:
            edges_filename (str): path to a txt-file with edges
            identifiers_filename (str): path to a txt-file with node identifiers

        Output:
            contact_network (ContactNetwork): initialized object
        """
        edges       = cls.__load_edges_from(edges_filename)
        node_groups = cls.__load_node_groups_from(identifiers_filename)

        return cls(edges, node_groups)

    def __init__(
            self,
            edges,
            node_groups):
        """
        Constructor

        Input:
            edges (np.array): (n_edges,2) array of edges
            node_groups (dict): a map from identifier indices to arrays of nodes
        """
        upper_tri_edges = self.__only_upper_triangular(edges)
        nodes = np.unique(upper_tri_edges)

        # in the following, first enforce the ascending order of the nodes,
        # then add edges, and then weed out missing labels (for example, there
        # might be no node '0', so every node 'j' gets mapped to 'j-1', and the
        # edges are remapped accordingly)
        #
        # this whole workaround is needed so that we can then simply say that
        # nodes 0--40, for instance, are health-care workers (instead of dealing
        # with permutations and such)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(upper_tri_edges)
        self.graph = nx.convert_node_labels_to_integers(self.graph,
                                                        ordering='sorted')
        self.__check_correct_format()

        self.node_groups = node_groups

    @staticmethod
    def __load_edges_from(filename):
        """
        Load edges from a txt-file

        Input:
            filename (str): path to a txt-file with edges

        Output:
            edges (np.array): (n_edges,2) array of edges
        """
        edges = np.loadtxt(filename, dtype=int, comments='#')
        return edges

    @staticmethod
    def __load_node_groups_from(filename):
        """
        Load node groups from a txt-file

        Input:
            filename (str): path to a txt-file with a node-to-identifier map

        Output:
            node_groups (dict): a map from identifier indices to arrays of nodes
        """
        nodes_and_identifiers = np.loadtxt(filename, dtype=str)

        nodes       = nodes_and_identifiers[:,0].astype(np.int)
        identifiers = nodes_and_identifiers[:,1]

        health_workers = nodes[identifiers == ContactNetwork.health_workers_id]
        community      = nodes[identifiers == ContactNetwork.community_id]

        node_groups = {
                ContactNetwork.health_workers_index : health_workers,
                ContactNetwork.community_index      : community }

        return node_groups

    def __only_upper_triangular(
            self,
            edges):
        """
        Filter out lower-triangular nodes, leaving upper-triangular only

        Input:
            edges (np.array): (n_edges,2) array of edges

        Output:
            edges (np.array): (L,2) array of upper-triangular edges
        """
        upper_tri_edges_mask = edges[:,0] < edges[:,1] # a boolean array
        return edges[upper_tri_edges_mask]

    def __check_correct_format(self):
        """
        Check whether the graph is in the correct format

        The following is checked:
            - all nodes are integers in the range 0..(n-1)
            - nodes are sorted in ascending order

        Output:
            None
        """
        nodes = self.graph.nodes
        node_count = self.get_node_count()
        if not np.array_equal(nodes, np.arange(node_count)):
            raise ValueError(
                    self.__class__.__name__
                    + ": graph format is incorrect")

    def __convert_array_to_dict(
            self,
            values_array):
        """
        Convert numpy array to dictionary with node indices as keys

        Input:
            values_array (np.array): (n_nodes,) array of values

        Output:
            values_dict (dict): a mapping node -> value
        """
        return { node: values_array[node] for node in self.get_nodes() }

    def get_health_workers(self):
        """
        Get health worker nodes

        Output:
            health_workers (np.array): (K,) array of node indices
        """
        return self.node_groups[ContactNetwork.health_workers_index]

    def get_community(self):
        """
        Get community nodes

        Output:
            community (np.array): (K,) array of node indices
        """
        return self.node_groups[ContactNetwork.community_index]

    def get_node_count(self):
        """
        Get the total number of nodes

        Output:
            n_nodes (int): total number of nodes
        """
        return self.graph.number_of_nodes()

    def get_edge_count(self):
        """
        Get the total number of edges

        Output:
            n_edges (int): total number of edges
        """
        return self.graph.number_of_edges()

    # TODO hide implementation, expose interfaces (i.e. delete get_graph)
    def get_graph(self):
        """
        Get the graph

        Output:
            graph (nx.Graph): graph object with node and edge attributes
        """
        return self.graph

    def get_nodes(self):
        """
        Get all nodes of the graph

        Output:
            nodes (np.array): (n_nodes,) array of node indices
        """
        return np.arange(self.get_node_count())

    def get_edges(self):
        """
        Get all edges of the graph

        Output:
            edges (np.array): (n_edges,2) array of pairs of node indices
        """
        return np.array(self.graph.edges)

    def get_incident_edges(
            self,
            node):
        """
        Get incident edges of a node

        Input:
            node (int): node whose incident edges to retrieve

        Output:
            edges (list): list of tuples, each of which is an incident edge
        """
        return list(self.graph.edges(node))

    def get_edge_weights(self):
        """
        Get edge weights of the graph as a scipy.sparse matrix

        Output:
            edge_weights (scipy.sparse.csr.csr_matrix): adjacency matrix
        """
        return nx.to_scipy_sparse_matrix(self.graph, weight=ContactNetwork.WJI)

    def get_age_groups(self):
        """
        Get the age groups of the nodes

        Output:
            age_groups (np.array): (n_nodes,) array of age groups
        """
        age_groups_dict = nx.get_node_attributes(
            self.graph, name=ContactNetwork.age_group)
        return np.fromiter(age_groups_dict.values(), dtype=int)

    def get_lambdas(self):
        """
        Get λ_min and λ_max attributes of the nodes

        Output:
            λ_min (np.array): (n_nodes,) array of values
            λ_max (np.array): (n_nodes,) array of values
        """
        λ_min_dict = nx.get_node_attributes(
            self.graph, name=ContactNetwork.lambda_min)
        λ_max_dict = nx.get_node_attributes(
            self.graph, name=ContactNetwork.lambda_max)
        return (np.fromiter(λ_min_dict.values(), dtype=float),
                np.fromiter(λ_max_dict.values(), dtype=float))

    def set_lambdas(
            self,
            λ_min,
            λ_max):
        """
        Set λ_min and λ_max attributes to the nodes

        Input:
            λ_min (int),
                  (float): constant value to be assigned to all nodes
                  (dict): a mapping node -> value
                  (np.array): (n_nodes,) array of values
            λ_max (int),
                  (float): constant value to be assigned to all nodes
                  (dict): a mapping node -> value
                  (np.array): (n_nodes,) array of values

        Output:
            None
        """
        self.set_lambda_min(λ_min)
        self.set_lambda_max(λ_max)

    def set_lambda_min(
            self,
            λ_min):
        """
        Set λ_min attribute to the nodes

        Input:
            λ_min (int),
                  (float): constant value to be assigned to all nodes
                  (dict): a mapping node -> value
                  (np.array): (n_nodes,) array of values

        Output:
            None
        """
        self.__set_node_attributes(λ_min, ContactNetwork.lambda_min)

    def set_lambda_max(
            self,
            λ_max):
        """
        Set λ_max attribute to the nodes

        Input:
            λ_max (int),
                  (float): constant value to be assigned to all nodes
                  (dict): a mapping node -> value
                  (np.array): (n_nodes,) array of values

        Output:
            None
        """
        self.__set_node_attributes(λ_max, ContactNetwork.lambda_max)

    def __set_node_attributes(
            self,
            values,
            name):
        """
        Set node attributes of the graph by name

        Input:
            values (int),
                   (float): constant value to be assigned to all nodes
                   (dict): a mapping node -> value
                   (np.array): (n_nodes,) array of values
            name (str): name of the attributes

        Output:
            None
        """
        if isinstance(values, (int, float, dict)):
            self.__set_node_attributes_const_dict(values, name)
        elif isinstance(values, np.ndarray):
            self.__set_node_attributes_array(values, name)
        else:
            raise ValueError(
                    self.__class__.__name__
                    + ": this type of argument is not supported: "
                    + values.__class__.__name__)

    def __set_node_attributes_const_dict(
            self,
            values,
            name):
        """
        Set node attributes of the graph by name

        Input:
            values (int),
                   (float): constant value to be assigned to all nodes
                   (dict): a mapping node -> value
            name (str): name of the attributes

        Output:
            None
        """
        nx.set_node_attributes(self.graph, values=values, name=name)

    def __set_node_attributes_array(
            self,
            values,
            name):
        """
        Set node attributes of the graph by name

        Input:
            values (np.array): (n_nodes,) array of values
            name (str): name of the attributes

        Output:
            None
        """
        values_dict = self.__convert_array_to_dict(values)
        self.__set_node_attributes_const_dict(values_dict, name)

    def set_transition_rates(
            self,
            transition_rates):
        """
        Set transitions rates (E->I etc.) as attributes of the nodes

        Input:
            transition_rates (TransitionRates): object with instance variables:
                exposed_to_infected
                infected_to_hospitalized
                infected_to_resistant
                infected_to_deceased
                hospitalized_to_resistant
                hospitalized_to_deceased
        Output:
            None
        """
        self.__set_node_attributes(
                transition_rates.exposed_to_infected,
                ContactNetwork.e_to_i)
        self.__set_node_attributes(
                transition_rates.infected_to_hospitalized,
                ContactNetwork.i_to_h)
        self.__set_node_attributes(
                transition_rates.infected_to_resistant,
                ContactNetwork.i_to_r)
        self.__set_node_attributes(
                transition_rates.infected_to_deceased,
                ContactNetwork.i_to_d)
        self.__set_node_attributes(
                transition_rates.hospitalized_to_resistant,
                ContactNetwork.h_to_r)
        self.__set_node_attributes(
                transition_rates.hospitalized_to_deceased,
                ContactNetwork.h_to_d)

    def set_edge_weights(
            self,
            edge_weights):
        """
        Set edge weights of the graph

        Input:
            edge_weights (dict): a mapping edge -> weight
        Output:
            None
        """
        nx.set_edge_attributes(
                self.graph, values=edge_weights, name=ContactNetwork.WJI)

    def add_edges(
            self,
            edges):
        """
        Add edges to the graph

        Input:
            edges (list): list of tuples, each of which is an edge
        Output:
            None
        """
        self.graph.add_edges_from(edges)

    def remove_edges(
            self,
            edges):
        """
        Remove edges from the graph

        Input:
            edges (list): list of tuples, each of which is an edge
        Output:
            None
        """
        self.graph.remove_edges_from(edges)

    def __draw_from(
            self,
            distribution):
        """
        Draw from distribution an array of length equal to the node count

        Input:
            distribution (np.array): discrete distribution (should sum to 1)

        Output:
            age_groups (np.array): (n_nodes,) array of age groups
        """
        n_nodes = self.get_node_count()
        n_groups = len(distribution)
        age_groups = np.random.choice(n_groups, p=distribution, size=n_nodes)

        return age_groups

    def draw_and_set_age_groups(
            self,
            distribution):
        """
        Draw from `distribution` and set age groups to the nodes

        Input:
            distribution (np.array): discrete distribution (should sum to 1)

        Output:
            None
        """
        age_groups = self.__draw_from(distribution)
        self.__set_node_attributes(age_groups, ContactNetwork.age_group)

    def isolate(
            self,
            sick_nodes,
            λ_isolation=1.0):
        """
        Isolate sick nodes by setting λ's of sick nodes to λ_isolation

        Input:
            sick_nodes (np.array): (n_sick,) array of indices of sick nodes

        Output:
            None
        """
        (λ_min, λ_max) = self.get_lambdas()

        λ_min[sick_nodes] = λ_isolation
        λ_max[sick_nodes] = λ_isolation

        self.set_lambdas(λ_min, λ_max)

    def build_user_network_using(
            self,
            user_graph_builder):
        """
        Build user network using provided builder

        Input:
            user_graph_builder (callable): an object to build user_graph

        Output:
            user_network (ContactNetwork): built user network
        """
        user_graph = user_graph_builder(self.graph)
        return ContactNetwork.from_networkx_graph(user_graph)

    # TODO extract into a separate class
    def generate_diagram_indep(self):
        """
        Generate diagram with independent transition rates

        Output:
            diagram_indep (nx.DiGraph): diagram with independent rates
        """
        diagram_indep = nx.DiGraph()
        diagram_indep.add_node('S')
        diagram_indep.add_edge(
                'E', 'I', rate=1, weight_label=ContactNetwork.e_to_i)
        diagram_indep.add_edge(
                'I', 'H', rate=1, weight_label=ContactNetwork.i_to_h)
        diagram_indep.add_edge(
                'I', 'R', rate=1, weight_label=ContactNetwork.i_to_r)
        diagram_indep.add_edge(
                'I', 'D', rate=1, weight_label=ContactNetwork.i_to_d)
        diagram_indep.add_edge(
                'H', 'R', rate=1, weight_label=ContactNetwork.h_to_r)
        diagram_indep.add_edge(
                'H', 'D', rate=1, weight_label=ContactNetwork.h_to_d)
        return diagram_indep

    # TODO extract into a separate class
    def generate_diagram_neigh(
            self,
            community_rate,
            hospital_rate):
        """
        Generate diagram with transmition rates that depend on neighbors

        Input:
            community_rate (float): rate at which infected infect susceptible
            hospital_rate (float): rate at which hospitalized infect susceptible

        Output:
            diagram_neigh (nx.DiGraph): diagram with neighbor-dependent rates
        """
        diagram_neigh = nx.DiGraph()
        diagram_neigh.add_edge(
                ('I', 'S'),
                ('I', 'E'),
                rate=community_rate,
                weight_label=ContactNetwork.WJI)
        diagram_neigh.add_edge(
                ('H', 'S'),
                ('H', 'E'),
                rate=hospital_rate,
                weight_label=ContactNetwork.WJI)
        return diagram_neigh


