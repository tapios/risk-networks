from functools import singledispatchmethod

import numpy as np
import scipy.sparse as scspa
import networkx as nx

class ContactNetwork:
    """
    Store and mutate a contact network
    """

    HOSPITAL_BEDS_ID  = 'HOSP'
    HEALTH_WORKERS_ID = 'HCW'
    COMMUNITY_ID      = 'CITY'
    HOSPITAL_BEDS_INDEX  = 0
    HEALTH_WORKERS_INDEX = 1
    COMMUNITY_INDEX      = 2

    # TODO change the string constants
    AGE_GROUP = 'age_group'
    LAMBDA_MIN = 'night_inception_rate'
    LAMBDA_MAX = 'day_inception_rate'

    BETA = 'exposed_by_infected'
    BETA_PRIME = 'exposed_by_hospitalized'

    E_TO_I = 'exposed_to_infected'
    I_TO_H = 'infected_to_hospitalized'
    I_TO_R = 'infected_to_resistant'
    I_TO_D = 'infected_to_deceased'
    H_TO_R = 'hospitalized_to_resistant'
    H_TO_D = 'hospitalized_to_deceased'

    def __init__(
            self,
            edges_filename,
            identifiers_filename):
        """
        Constructor

        Input:
            edges_filename (str): path to a txt-file with edges
            identifiers_filename (str): path to a txt-file with node identifiers
        """
        edges = self.__load_edges_from(edges_filename)
        upper_tri_edges = self.__only_upper_triangular(edges)

        self.graph = nx.Graph()
        self.graph.add_edges_from(upper_tri_edges)
        self.graph = nx.convert_node_labels_to_integers(self.graph)
        self.__check_correct_format()

        self.node_groups = self.__load_node_groups_from(identifiers_filename)

    def __load_edges_from(
            self,
            filename):
        """
        Load edges from a txt-file

        Input:
            filename (str): path to a txt-file with edges

        Output:
            edges (np.array): (n_edges,2) array of edges
        """
        edges = np.loadtxt(filename, dtype=int, comments='#')
        return edges

    def __load_node_groups_from(
            self,
            filename):
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

        hospital_beds  = nodes[identifiers == ContactNetwork.HOSPITAL_BEDS_ID]
        health_workers = nodes[identifiers == ContactNetwork.HEALTH_WORKERS_ID]
        community      = nodes[identifiers == ContactNetwork.COMMUNITY_ID]

        node_groups = {
                ContactNetwork.HOSPITAL_BEDS_INDEX  : hospital_beds,
                ContactNetwork.HEALTH_WORKERS_INDEX : health_workers,
                ContactNetwork.COMMUNITY_INDEX      : community }

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
        return self.node_groups[ContactNetwork.HEALTH_WORKERS_INDEX]

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
        return nx.to_scipy_sparse_matrix(self.graph, weight=ContactNetwork.BETA)

    def get_age_groups(self):
        """
        Get the age groups of the nodes

        Output:
            age_groups (np.array): (n_nodes,) array of age groups
        """
        age_groups_dict = nx.get_node_attributes(
            self.graph, name=ContactNetwork.AGE_GROUP)
        return np.fromiter(age_groups_dict.values(), dtype=int)

    def get_lambdas(self):
        """
        Get λ_min and λ_max attributes of the nodes

        Output:
            λ_min (np.array): (n_nodes,) array of values
            λ_max (np.array): (n_nodes,) array of values
        """
        λ_min_dict = nx.get_node_attributes(
            self.graph, name=ContactNetwork.LAMBDA_MIN)
        λ_max_dict = nx.get_node_attributes(
            self.graph, name=ContactNetwork.LAMBDA_MAX)
        return (np.fromiter(λ_min_dict.values(), dtype=float),
                np.fromiter(λ_max_dict.values(), dtype=float))

    # TODO: unfortunately, dispatch works on the first argument only; thus, the
    # current code can introduce an undesired behavior when
    #     type(λ_min) == int and type(λ_max) == np.ndarray
    # Possible solutions (1st one is preferred):
    #   - not dispatch at the `set_lambdas` level, but instead implement a
    #     helper method that dispatches on each λ individually;
    #   - simply have an if inside since there are really only two different
    #     behaviors: one for int, float, dict, and one for np.array
    @singledispatchmethod
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
        raise ValueError(
                self.__class__.__name__
                + ": this type of argument is not supported: "
                + λ_min.__class__.__name__
                + ", "
                + λ_max.__class__.__name__)

    @set_lambdas.register(int)
    @set_lambdas.register(float)
    @set_lambdas.register(dict)
    def set_lambdas_const_dict(
            self,
            λ_min,
            λ_max):
        """
        Set λ_min and λ_max attributes to the nodes

        Input:
            λ_min (int),
                  (float): constant value to be assigned to all nodes
                  (dict): a mapping node -> value
            λ_max (int),
                  (float): constant value to be assigned to all nodes
                  (dict): a mapping node -> value

        Output:
            None
        """
        nx.set_node_attributes(
                self.graph, values=λ_min, name=ContactNetwork.LAMBDA_MIN)
        nx.set_node_attributes(
                self.graph, values=λ_max, name=ContactNetwork.LAMBDA_MAX)

    @set_lambdas.register(np.ndarray)
    def set_lambdas_array(
            self,
            λ_min,
            λ_max):
        """
        Set λ_min and λ_max attributes to the nodes

        Input:
            λ_min (np.array): (n_nodes,) array of values
            λ_max (np.array): (n_nodes,) array of values

        Output:
            None
        """
        λ_min_dict = self.__convert_array_to_dict(λ_min)
        λ_max_dict = self.__convert_array_to_dict(λ_max)
        self.set_lambdas_const_dict(λ_min_dict, λ_max_dict)

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
        nx.set_node_attributes(
                self.graph,
                values=transition_rates.exposed_to_infected,
                name=ContactNetwork.E_TO_I)
        nx.set_node_attributes(
                self.graph,
                values=transition_rates.infected_to_hospitalized,
                name=ContactNetwork.I_TO_H)
        nx.set_node_attributes(
                self.graph,
                values=transition_rates.infected_to_resistant,
                name=ContactNetwork.I_TO_R)
        nx.set_node_attributes(
                self.graph,
                values=transition_rates.infected_to_deceased,
                name=ContactNetwork.I_TO_D)
        nx.set_node_attributes(
                self.graph,
                values=transition_rates.hospitalized_to_resistant,
                name=ContactNetwork.H_TO_R)
        nx.set_node_attributes(
                self.graph,
                values=transition_rates.hospitalized_to_deceased,
                name=ContactNetwork.H_TO_D)

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
                self.graph, values=edge_weights, name=ContactNetwork.BETA)
        nx.set_edge_attributes(
                self.graph, values=edge_weights, name=ContactNetwork.BETA_PRIME)

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
        age_groups_array = self.__draw_from(distribution)
        age_groups_dict  = self.__convert_array_to_dict(age_groups_array)

        nx.set_node_attributes(
            self.graph, values=age_groups_dict, name=ContactNetwork.AGE_GROUP)

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
                'E', 'I', rate=1, weight_label=ContactNetwork.E_TO_I)
        diagram_indep.add_edge(
                'I', 'H', rate=1, weight_label=ContactNetwork.I_TO_H)
        diagram_indep.add_edge(
                'I', 'R', rate=1, weight_label=ContactNetwork.I_TO_R)
        diagram_indep.add_edge(
                'I', 'D', rate=1, weight_label=ContactNetwork.I_TO_D)
        diagram_indep.add_edge(
                'H', 'R', rate=1, weight_label=ContactNetwork.H_TO_R)
        diagram_indep.add_edge(
                'H', 'D', rate=1, weight_label=ContactNetwork.H_TO_D)
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
                weight_label=ContactNetwork.BETA)
        diagram_neigh.add_edge(
                ('H', 'S'),
                ('H', 'E'),
                rate=hospital_rate,
                weight_label=ContactNetwork.BETA_PRIME)
        return diagram_neigh


