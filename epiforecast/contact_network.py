import numpy as np
import scipy.sparse as scspa
import networkx as nx

class ContactNetwork:
    """
    Store and mutate a contact network
    """

    HEALTH_WORKERS_ID = 'HCW'
    COMMUNITY_ID      = 'CITY'
    HEALTH_WORKERS_INDEX = 0
    COMMUNITY_INDEX      = 1

    # TODO extract into Glossary class
    AGE_GROUP = 'age_group'

    LAMBDA_MIN = 'minimum_contact_rate'
    LAMBDA_MAX = 'maximum_contact_rate'

    WJI = 'edge_weights'

    E_TO_I = 'exposed_to_infected'
    I_TO_H = 'infected_to_hospitalized'
    I_TO_R = 'infected_to_resistant'
    I_TO_D = 'infected_to_deceased'
    H_TO_R = 'hospitalized_to_resistant'
    H_TO_D = 'hospitalized_to_deceased'

    @classmethod
    def from_networkx_graph(
            cls,
            graph,
            convert_labels_to_0N=True):
        """
        Create an object from a nx.Graph object

        Input:
            graph (nx.Graph): an object to use as a contact network graph
            convert_labels_to_0N (boolean): convert node labels to 0..N-1

        Output:
            contact_network (ContactNetwork): initialized object
        """
        edges       = np.array(graph.edges)
        node_groups = {
                ContactNetwork.HEALTH_WORKERS_INDEX : np.array([]),
                ContactNetwork.COMMUNITY_INDEX      : np.array(graph.nodes) }

        return cls(edges, node_groups, convert_labels_to_0N)

    @classmethod
    def from_files(
            cls,
            edges_filename,
            identifiers_filename,
            convert_labels_to_0N=True):
        """
        Create an object from files that contain edges and identifiers

        Input:
            edges_filename (str): path to a txt-file with edges
            identifiers_filename (str): path to a txt-file with node identifiers
            convert_labels_to_0N (boolean): convert node labels to 0..N-1

        Output:
            contact_network (ContactNetwork): initialized object
        """
        edges       = cls.__load_edges_from(edges_filename)
        node_groups = cls.__load_node_groups_from(identifiers_filename)

        return cls(edges, node_groups, convert_labels_to_0N)

    def __init__(
            self,
            edges,
            node_groups,
            convert_labels_to_0N):
        """
        Constructor

        Input:
            edges (np.array): (n_edges,2) array of edges
            node_groups (dict): a map from identifier indices to arrays of nodes
            convert_labels_to_0N (boolean): convert node labels to 0..N-1
        """
        nodes = np.unique(edges)

        # in the following, first enforce the ascending order of the nodes,
        # then add edges, and then (possibly) weed out missing labels (for
        # example, there might be no node '0', so every node 'j' gets mapped to
        # 'j-1', and the edges are remapped accordingly)
        #
        # this whole workaround is needed so that we can then simply say that
        # nodes 0..40, for instance, are health-care workers (instead of dealing
        # with permutations and such)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        if convert_labels_to_0N:
            self.graph = nx.convert_node_labels_to_integers(self.graph,
                                                            ordering='sorted')
        self.__check_correct_format(convert_labels_to_0N)

        self.node_groups = node_groups

        # set default attributes to 1.0 in the case of a static network, where
        # contact_simulator is not called (otherwise they are implicitly set)
        self.set_edge_weights(1.0)

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

        health_workers = nodes[identifiers == ContactNetwork.HEALTH_WORKERS_ID]
        community      = nodes[identifiers == ContactNetwork.COMMUNITY_ID]

        node_groups = {
                ContactNetwork.HEALTH_WORKERS_INDEX : health_workers,
                ContactNetwork.COMMUNITY_INDEX      : community }

        return node_groups

    def __check_correct_format(
            self,
            check_labels_are_0N):
        """
        Check whether the graph is in the correct format

        The following is checked:
            - nodes are sorted in ascending order
        If `check_labels_are_0N` is true then also check
            - all nodes are integers in the range 0..N-1

        Input:
            check_labels_are_0N (boolean): check that node labels are 0..N-1

        Output:
            None
        """
        correct_format = True

        nodes = self.get_nodes()
        if not np.all(nodes[:-1] <= nodes[1:]): # if not "ascending order"
            correct_format = False

        if check_labels_are_0N:
            node_count = self.get_node_count()
            if not np.array_equal(nodes, np.arange(node_count)):
                correct_format = False

        if not correct_format:
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

    def get_community(self):
        """
        Get community nodes

        Output:
            community (np.array): (K,) array of node indices
        """
        return self.node_groups[ContactNetwork.COMMUNITY_INDEX]

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
        return np.array(self.graph.nodes)

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
        self.__set_node_attributes(λ_min, ContactNetwork.LAMBDA_MIN)

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
        self.__set_node_attributes(λ_max, ContactNetwork.LAMBDA_MAX)

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

    def set_transition_rates_for_kinetic_model(
            self,
            transition_rates):
        """
        Set transition rates (exposed to infected etc.) as node attributes

        Note: these transition rates are only intended to be used by
        KineticModel; hence, this method does not really belong here, and should
        be implemented in KineticModel instead

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
                ContactNetwork.E_TO_I)
        self.__set_node_attributes(
                transition_rates.infected_to_hospitalized,
                ContactNetwork.I_TO_H)
        self.__set_node_attributes(
                transition_rates.infected_to_resistant,
                ContactNetwork.I_TO_R)
        self.__set_node_attributes(
                transition_rates.infected_to_deceased,
                ContactNetwork.I_TO_D)
        self.__set_node_attributes(
                transition_rates.hospitalized_to_resistant,
                ContactNetwork.H_TO_R)
        self.__set_node_attributes(
                transition_rates.hospitalized_to_deceased,
                ContactNetwork.H_TO_D)

    def set_edge_weights(
            self,
            edge_weights):
        """
        Set edge weights of the graph

        Input:
            edge_weights (int),
                         (float): constant value to be assigned to all edges
                         (dict): a mapping edge -> weight
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
        self.__set_node_attributes(age_groups, ContactNetwork.AGE_GROUP)

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
        return ContactNetwork.from_networkx_graph(user_graph, False)

    # TODO extract into a separate class
    @staticmethod
    def generate_diagram_indep():
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
    @staticmethod
    def generate_diagram_neigh(
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


