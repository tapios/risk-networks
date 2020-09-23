import numpy as np
import copy

from epiforecast.ensemble_adjustment_kalman_filter import EnsembleAdjustmentKalmanFilter

class DataAssimilator:
    """
    Collect and store observations, and provide DA updates given state and data
    """
    def __init__(
            self,
            observations,
            errors,
            *,
            n_assimilation_batches=1,
            transition_rates_to_update_str=None,
            transmission_rate_to_update_flag=None,
            update_type='global',
            full_svd=False,
            joint_cov_noise=1e-2,
            transition_rates_min=None,
            transition_rates_max=None,
            transmission_rate_min=None,
            transmission_rate_max=None):
        """
        Constructor

        Input:
            observations (list of Observation, [], Observation): observations
                    Generates the indices and covariances of observations.

            errors (list of Observation, [], Observation): error-checking observations
                    Error observations are used to compute online differences at
                    the observed (according to Errors) between Kinetic and
                    Master Equation models.

            n_assimilation_batches (int): number of random batches over which to
                                          assimilate
                    At the cost of information loss, one can batch assimilation
                    updates into random even-sized batches, the update scales
                    with O(num observation states^3) and is memory intensive.
                    Thus performing n x m-sized updates is far cheaper than an
                    nm-sized update.

            transition_rates_to_update_str (list of str): which rates to update
                    Must coincide with naming found in TransitionRates.
                    If not provided, will set to [].

            transmission_rate_to_update_flag (bool): whether to update transmission rate
                    If not provided will set False.

            update_type (str): how to perform updates
                    Three values are supported: 'global', 'local', 'neighbor'.

            full_svd (bool): whether to use full or reduced second SVD in EAKF

            joint_cov_noise (float): Tikhonov-regularization noise
        """
        if not isinstance(observations, list):
            observations = [observations]

        if not isinstance(errors, list):
            errors = [errors]

        self.observations = observations
        self.online_emodel = errors # online evaluations of errors

        self.n_assimilation_batches = n_assimilation_batches

        if transition_rates_to_update_str is None:
            transition_rates_to_update_str = []

        if transmission_rate_to_update_flag is None:
            transmission_rate_to_update_flag = False

        if not isinstance(transition_rates_to_update_str, list):
            transition_rates_to_update_str = [transition_rates_to_update_str]

        self.transition_rates_to_update_str = transition_rates_to_update_str
        self.transmission_rate_to_update_flag = transmission_rate_to_update_flag

        self.update_type = update_type

        if full_svd:
            self.damethod = EnsembleAdjustmentKalmanFilter(
                    prior_svd_reduced=True,
                    observation_svd_regularized=False,
                    joint_cov_noise=joint_cov_noise)
        else:
            self.damethod = EnsembleAdjustmentKalmanFilter(
                    prior_svd_reduced=True,
                    joint_cov_noise=joint_cov_noise)

        # storage for observations time : obj 
        self.stored_observed_states = {}
        self.stored_observed_nodes = {}
        self.stored_observed_means = {}
        self.stored_observed_variances = {}

        # range of transition rates
        self.transition_rates_min = transition_rates_min
        self.transition_rates_max = transition_rates_max 

        # range of transmission rate
        self.transmission_rate_min = transmission_rate_min 
        self.transmission_rate_max = transmission_rate_max 


    def find_observation_states(
            self,
            user_network,
            ensemble_state,
            data,
            current_time,
            verbose=False):
        """
        Make all the observations in the list self.observations.

        This sets observation.obs_states.

        Input:
            ...
            verbose (bool): whether to print observation name and states
        """
        if verbose:
            print("[ Data assimilator ]",
                  "Observation type : Number of Observed states")

        if current_time in self.stored_observed_states:
            observed_states = self.stored_observed_states[current_time]
            observed_nodes  = self.stored_observed_nodes[current_time]
        else:
            observed_states_list = []
            for observation in self.observations:
                observation.find_observation_states(user_network,
                                                    ensemble_state,
                                                    data)
                if observation.obs_states.size > 0:
                    observed_states_list.extend(observation.obs_states)
                    if verbose:
                        print("[ Data assimilator ]",
                              observation.name,
                              ":",
                              len(observation.obs_states))

            n_user_nodes = user_network.get_node_count()
            observed_states = np.unique(observed_states_list)
            observed_nodes  = np.unique(observed_states % n_user_nodes)

            self.stored_observed_states[current_time] = observed_states
            self.stored_observed_nodes[current_time]  = observed_nodes

        return observed_states, observed_nodes

    def observe(
            self,
            user_network,
            state,
            data,
            current_time,
            scale=None,
            noisy_measurement=True):

        if current_time in self.stored_observed_means:
            observed_means = self.stored_observed_means[current_time]
            observed_variances = self.stored_observed_variances[current_time]
            return observed_means, observed_variances
        
        else:
            observed_means = []
            observed_variances = []
            for observation in self.observations:
                if (observation.obs_states.size >0):
                    observation.observe(user_network,
                                        state,
                                        data,
                                        scale)

                    observed_means.extend(observation.mean)
                    observed_variances.extend(observation.variance)

            observed_means = np.array(observed_means)
            observed_variances = np.array(observed_variances)
            self.stored_observed_means[current_time] = observed_means
            self.stored_observed_variances[current_time] = observed_variances
            return observed_means, observed_variances

    def find_and_store_observations(
            self,
            ensemble_state,
            data,
            user_network,
            current_time,
            scale=None,
            noisy_measurement=True,
            verbose=False):
        self.find_observation_states(user_network,
                                     ensemble_state,
                                     data,
                                     current_time,
                                     verbose)

        self.observe(user_network,
                     ensemble_state,
                     data,
                     current_time)

    def compute_update_indices(
            self,
            user_network,
            obs_states,
            obs_nodes):
        """
        Compute state vector indices for DA update according to the update type

        Input:
            user_network (ContactNetwork): user network
            obs_states (np.array): (n_obs_states,) array of state indices
            obs_nodes (np.array): (n_obs_nodes,) array of node indices
        Output:
            update_states (np.array): (k,) array of state indices
        """
        if self.update_type == 'global':
            n_user_nodes = user_network.get_node_count()
            update_states = self.__compute_global_update_indices(
                    n_user_nodes,
                    obs_states)
        elif self.update_type == 'neighbor':
            update_states = self.__compute_neighbor_update_indices(
                    user_network,
                    obs_states,
                    obs_nodes)
        elif self.update_type == 'local':
            update_states = obs_states

        return update_states

    def __compute_global_update_indices(
            self,
            n_user_nodes,
            obs_states):
        """
        Compute state vector indices for DA update when performing global update

        This method returns state vector indices that correspond to the 'global'
        type of update, i.e. if there's at least one state from a compartment,
        that whole compartment should be updated.

        Input:
            n_user_nodes (int): total number of nodes in user_network
            obs_states (np.array): (n_obs_states,) array of state indices
        Output:
            update_states (np.array): (k,) array of state indices
        """
        compartment_indices = np.arange(n_user_nodes)

        update_compartments = np.unique( obs_states // n_user_nodes )
        update_states_2d = np.add.outer(n_user_nodes * update_compartments,
                                        compartment_indices)

        return update_states_2d.ravel()

    def __compute_neighbor_update_indices(
            self,
            user_network,
            obs_states,
            obs_nodes):
        """
        Compute state vector indices for DA update when performing neighbor update

        This method returns state vector indices that correspond to the
        'neighbor' type of update, i.e. first, adjacent to 'obs_nodes' nodes are
        added to the set of updated nodes; then, if there's at least one state
        from a compartment, that compartment is added to the set of updated
        compartments; finally, state vector indices that should be updated are
        the ones from each updated compartment for all updated nodes.

        Input:
            user_network (ContactNetwork): user network
            obs_states (np.array): (n_obs_states,) array of state indices
            obs_nodes (np.array): (n_obs_nodes,) array of node indices
        Output:
            update_states (np.array): (k,) array of state indices
        """
        neighbor_nodes = user_network.get_neighbors(obs_nodes)
        update_nodes = np.union1d(neighbor_nodes, obs_nodes)

        n_user_nodes = user_network.get_node_count()
        update_compartments = np.unique( obs_states // n_user_nodes )

        update_states_2d = np.add.outer(n_user_nodes * update_compartments,
                                        update_nodes)

        return update_states_2d.ravel()

    def generate_state_observation_operator(
            self,
            obs_states,
            update_states):
        """
        Generate the state observation operator (a.k.a. H_obs)

        Input:
            obs_states (np.array): (n_obs_states,) array of state indices
            update_states (np.array): (n_update_states,) array of state indices
        Output:
            H_obs (np.array): (n_obs_states, n_update_states) array of {0,1}
                              values
        """
        # XXX why was this obs_nodes.size, not obs_states.size?
        H_obs = np.zeros([obs_states.size, update_states.size])

        # obs_states is always a subset of update_states;
        # H_obs is (obs_states.size, update_states.size) matrix;
        # hence, the following mask is used to find correct indices
        # of obs_states when viewed as a subset of update_states
        obs_states_in_update_states_mask = np.isin(update_states,
                                                   obs_states)
        H_obs[np.arange(obs_states.size),
              obs_states_in_update_states_mask] = 1

        return H_obs

    # ensemble_state np.array([ensemble size, num status * num nodes]
    # data np.array([num status * num nodes])
    # contact network networkx.graph (if provided)
    # full_ensemble_transition_rates list[ensemble size] of  TransitionRates objects from epiforecast.populations
    # full_ensemble_transmission_rate np.array([ensemble size])
    def update(
            self,
            ensemble_state,
            data,
            full_ensemble_transition_rates,
            full_ensemble_transmission_rate,
            user_network,
            current_time,
            verbose=False,
            print_error=False):
        """
        Input:
            ...
            verbose (bool): whether to print info about each observation
        """
        if len(self.observations) == 0: # no update is performed; return input
            return ensemble_state, full_ensemble_transition_rates, full_ensemble_transmission_rate

        else:
            # Load states to observe
            obs_states = self.stored_observed_states[current_time]
            obs_nodes = self.stored_observed_nodes[current_time]

            if (obs_states.size > 0):
                # extract only those rates which we wish to update with DA
                (ensemble_transition_rates,
                 ensemble_transmission_rate
                ) = self.extract_model_parameters_to_update(
                        full_ensemble_transition_rates,
                        full_ensemble_transmission_rate,
                        obs_nodes)

                if verbose:
                    print("[ Data assimilator ] Total states to be assimilated: ",
                          obs_states.size)
                    print("[ Data assimilator ] Total parameters to be assimilated: ",
                            ensemble_transition_rates.shape[1]
                          + ensemble_transmission_rate.shape[1])

                # Load the truth, variances of the observation(s)
                truth = self.stored_observed_means[current_time]
                var = self.stored_observed_variances[current_time]
                # Perform DA model update with ensemble_state: states, transition and transmission rates
                prev_ensemble_state = copy.deepcopy(ensemble_state)

                n_user_nodes = user_network.get_node_count()
                if self.n_assimilation_batches == 1:
                    #note cov is obs.size x obs.size in size
                    cov = np.diag(var)
                
                    update_states = self.compute_update_indices(
                            user_network,
                            obs_states,
                            obs_nodes)
                    H_obs = self.generate_state_observation_operator(
                            obs_states,
                            update_states)

                    (ensemble_state[:, update_states],
                     new_ensemble_transition_rates,
                     new_ensemble_transmission_rate
                    ) = self.damethod.update(ensemble_state[:, update_states],
                                             ensemble_transition_rates,
                                             ensemble_transmission_rate,
                                             truth,
                                             cov,
                                             H_obs,
                                             print_error=print_error)

                    if self.transmission_rate_to_update_flag:
                        # Clip transmission rate into a reasonable range
                        new_ensemble_transmission_rate = np.clip(new_ensemble_transmission_rate,
                                                                 self.transmission_rate_min,
                                                                 self.transmission_rate_max)

                        # Weighted-averaging based on ratio of observed nodes 
                        new_ensemble_transmission_rate = self.weighted_averaged_transmission_rate( \
                                new_ensemble_transmission_rate,
                                ensemble_transmission_rate,
                                n_user_nodes,
                                obs_nodes.size)

                else: # perform DA update in batches
                    if self.update_type == 'global':
                        raise NotImplementedError(
                                self.__class__.__name__
                                + ": batching is not implemented yet for: "
                                + "'global'")

                    permuted_idx = np.random.permutation(np.arange(obs_states.size))
                    batches = np.array_split(permuted_idx,
                                             self.n_assimilation_batches)

                    ensemble_size = ensemble_transition_rates.shape[0]
                    ensemble_transition_rates_reshaped = ensemble_transition_rates.reshape(
                            ensemble_size,
                            obs_nodes.shape[0],
                            -1)

                    for batch in batches:
                        batch.sort()
                        cov_batch = np.diag(var[batch])
                        if self.update_type == 'local':
                            update_states = obs_states[batch]
                            H_obs = np.eye(batch.size)
                        elif self.update_type == 'neighbor':
                            update_states = self.compute_update_indices(user_network,
                                    obs_states[batch],
                                    obs_nodes[batch])
                            H_obs = self.generate_state_observation_operator(
                                    obs_states[batch], 
                                    update_states)

                        (ensemble_state[:, update_states],
                         new_ensemble_transition_rates_batch,
                         new_ensemble_transmission_rate
                        ) = self.damethod.update(ensemble_state[:, update_states],
                                ensemble_transition_rates_reshaped[:,batch,:].reshape(ensemble_size,-1),
                                                 ensemble_transmission_rate,
                                                 truth[batch],
                                                 cov_batch,
                                                 H_obs,
                                                 print_error=print_error)

                        ensemble_transition_rates_reshaped[:,batch,:] = \
                        new_ensemble_transition_rates_batch.reshape(ensemble_size, batch.size, -1)

                        if self.transmission_rate_to_update_flag:
                            # Clip transmission rate into a reasonable range
                            new_ensemble_transmission_rate = np.clip(new_ensemble_transmission_rate,
                                                                     self.transmission_rate_min,
                                                                     self.transmission_rate_max)

                            # Weighted-averaging based on ratio of observed nodes 
                            new_ensemble_transmission_rate = self.weighted_averaged_transmission_rate( \
                                    new_ensemble_transmission_rate,
                                    ensemble_transmission_rate,
                                    n_user_nodes,
                                    batch.size)

                    new_ensemble_transition_rates = ensemble_transition_rates_reshaped.reshape(
                            ensemble_size,-1)

                if len(self.transition_rates_to_update_str) > 0:
                    new_ensemble_transition_rates = self.clip_transition_rates(new_ensemble_transition_rates,
                                                                               obs_nodes.size)

                self.sum_to_one(prev_ensemble_state, ensemble_state)

                # set the updated rates in the TransitionRates object and
                # return the full rates.
                (full_ensemble_transition_rates,
                 full_ensemble_transmission_rate
                ) = self.assign_updated_model_parameters(
                        new_ensemble_transition_rates,
                        new_ensemble_transmission_rate,
                        full_ensemble_transition_rates,
                        full_ensemble_transmission_rate,
                        obs_nodes)

                if print_error:
                    print("[ Data assimilator ] EAKF error:", self.damethod.error[-1])
            else:
                if verbose:
                    print("[ Data assimilator ] No assimilation required")

            # Error to truth
            if len(self.online_emodel)>0:
                self.error_to_truth_state(ensemble_state,data)

            # Return ensemble_state, transition rates, and transmission rate
            return ensemble_state, full_ensemble_transition_rates, full_ensemble_transmission_rate

    #defines a method to take a difference to the data state
    def error_to_truth_state(
            self,
            ensemble_state,
            data):

        em = self.online_emodel # get corresponding error model

        # Make sure you have a deterministic ERROR model - or it will not match truth
        # without further seeding
        em.make_new_observation(ensemble_state)

        predicted_infected = em.obs_states

        truth = data[np.arange(em.n_status*em.N)]

        #print(truth[predicted_infected])

        em.make_new_observation(truth[np.newaxis, :])

        # take error
        actual_infected= em.obs_states

        different_states=np.zeros(em.n_status*em.N)
        different_states[predicted_infected]=1.0
        different_states[actual_infected] = different_states[actual_infected]-1.0

        number_different=np.maximum(different_states,0.0)-np.minimum(different_states,0.0)

        print("[ Data assimilator ] Differences between predicted and true I>0.5:",
              np.sum(number_different).astype(int))


    #as we measure a subset of states, we may need to enforce other states to sum to one

    def sum_to_one(
            self,
            prev_ensemble_state,
            ensemble_state):

        N = self.observations[0].N


        # First obtain the mass contained in category "E"
        prev_tmp = prev_ensemble_state.reshape(prev_ensemble_state.shape[0], 5, N)
        Emass = 1.0 - np.sum(prev_tmp,axis=1) # E= 1 - (S + I + H + R + D)
        Emass = np.clip(Emass,0,1)
        # for each observation we get the observed status e.g 'I' and fix it
        # (as # it was updated); we then normalize the other states e.g
        # (S,'E',H,R,D) over the difference 1-I
        for observation in self.observations:
            if not observation.obs_states.size > 0:
                continue

            observed_nodes = np.remainder(observation.obs_states,N)
            updated_status = observation.obs_status_idx

            free_statuses = [ i for i in range(5) if i!= updated_status]
            tmp = ensemble_state.reshape(ensemble_state.shape[0], 5, N)

            # create arrays of the mass in the observed and the unobserved
            # "free" statuses at the observed nodes.
            observed_tmp = tmp[:,:,observed_nodes]
            updated_mass = observed_tmp[:, updated_status, :]
            free_states  = observed_tmp

            # remove this axis for the sum (but maintain the shape)
            free_states[:, updated_status, :] = np.zeros(
                    [free_states.shape[0], 1, free_states.shape[2]])

            free_mass = np.sum(free_states,axis=1) + Emass[:,observed_nodes]

            for i in free_statuses:
                #no_update_weight = (free_mass < 0.001)
                no_update_weight = (free_mass == 0)
                new_ensemble_state = (1.0 - updated_mass[:,0,:])\
                                   * (free_states[:, i, :] / np.maximum(1e-9,free_mass))
                ensemble_state[:, i*N+observed_nodes] = (no_update_weight) *  ensemble_state[:, i*N+observed_nodes]\
                                                      + (1-no_update_weight) * new_ensemble_state

    def extract_model_parameters_to_update(
            self,
            full_ensemble_transition_rates,
            full_ensemble_transmission_rate,
            obs_nodes):
        """
        Extract model parameters for update from lists into np.arrays

        Input:
            full_ensemble_transition_rates (list): list of TransitionRates from
                                                   which to extract rates for
                                                   update
            full_ensemble_transmission_rate (list): list of floats/ints
            obs_nodes (np.array): (m,) array of node indices

        Output:
            ensemble_transition_rates (np.array): (n_ensemble,k) array of values
            ensemble_transmission_rate (np.array): (n_ensemble,) array of values
        """
        n_ensemble = len(full_ensemble_transition_rates)

        # We extract only the transition rates we wish to be updated
        # stored as an [ensemble size x transition rates (to be updated)] np.array
        if len(self.transition_rates_to_update_str) > 0:
            for i, transition_rates in enumerate(full_ensemble_transition_rates):
                rates_member = []
                for rate_name in self.transition_rates_to_update_str:
                    # clinical_parameter is either a float or (n_user_nodes,) np.array
                    clinical_parameter = (
                            transition_rates.get_clinical_parameter(rate_name))

                    if isinstance(clinical_parameter, np.ndarray):
                        # only extract the observed values
                        clinical_parameter = clinical_parameter[obs_nodes]

                    rates_member.append(clinical_parameter)

                rates_member = np.hstack(rates_member)

                if i == 0:
                    ensemble_transition_rates = np.empty((0, rates_member.size),
                                                         dtype=float)
                ensemble_transition_rates = np.append(ensemble_transition_rates,
                                                      [rates_member],
                                                      axis=0)

            ensemble_transition_rates = np.vstack(ensemble_transition_rates)

        else: # set to column of empties
            ensemble_transition_rates = np.empty((n_ensemble, 0), dtype=float)

        if self.transmission_rate_to_update_flag:
            ensemble_transmission_rate = full_ensemble_transmission_rate
        else: # set to column of empties
            ensemble_transmission_rate = np.empty((n_ensemble, 0), dtype=float)

        return ensemble_transition_rates, ensemble_transmission_rate

    def assign_updated_model_parameters(
            self,
            new_ensemble_transition_rates,
            new_ensemble_transmission_rate,
            full_ensemble_transition_rates,
            full_ensemble_transmission_rate,
            obs_nodes):
        """
        Assign updated model parameters from np.arrays into corresponding lists

        Input:
            new_ensemble_transition_rates (np.array): (n_ensemble,k) array of
                                                      values
            new_ensemble_transmission_rate (np.array): (n_ensemble,) array of
                                                       values
            full_ensemble_transition_rates (list): list of TransitionRates, to
                                                   be updated
            full_ensemble_transmission_rate (list): list of floats/ints, to be
                                                    updated
            obs_nodes (np.array): (m,) array of node indices

        Output:
            full_ensemble_transition_rates (list): same object as in input
            full_ensemble_transmission_rate (list): same object as in input
        """
        if len(self.transition_rates_to_update_str) > 0:
            for transition_rates_from, transition_rates_to in zip(
                    new_ensemble_transition_rates,
                    full_ensemble_transition_rates):
                for rate_name in self.transition_rates_to_update_str:
                    # Need to go back from numpy array to setting rates
                    # We obtain the size, then update the corresponding transition rate
                    # Then delete this an move onto the next rate
                    clinical_parameter = (
                            transition_rates_to.get_clinical_parameter(
                                rate_name))
                    if isinstance(clinical_parameter, np.ndarray):
                        rate_size = obs_nodes.size
                        new_rates = clinical_parameter
                        new_rates[obs_nodes] = transition_rates_from[:rate_size]
                    else:
                        rate_size = 1
                        new_rates = transition_rates_from[0]

                    transition_rates_to.set_clinical_parameter(rate_name,
                                                               new_rates)
                    transition_rates_from = np.delete(transition_rates_from,
                                                      np.s_[:rate_size])

                transition_rates_to.calculate_from_clinical()

        if self.transmission_rate_to_update_flag:
            full_ensemble_transmission_rate = new_ensemble_transmission_rate

        return full_ensemble_transition_rates, full_ensemble_transmission_rate

    def clip_transition_rates(self, ensemble_transition_rates, obs_nodes_num):
        """
        Clip the values of tranistion rates into pre-defined ranges 

        Input:
            ensemble_transition_rates (np.array): (n_ensemble,k) array
            obs_nodes_num (int): number of observed nodes

        Output:
            ensemble_transition_rates (np.array): (n_ensemble,k) array
        """
        ensemble_size = ensemble_transition_rates.shape[0]
        ensemble_transition_rates = ensemble_transition_rates.reshape(ensemble_size,
                                                                      obs_nodes_num,
                                                                      -1)
        for i, transition_rates_str in enumerate(self.transition_rates_to_update_str):
            ensemble_transition_rates[:,:,i] = np.clip(ensemble_transition_rates[:,:,i],
                                                       self.transition_rates_min[transition_rates_str],
                                                       self.transition_rates_max[transition_rates_str])

        return ensemble_transition_rates.reshape(ensemble_size, -1)

    def weighted_averaged_transmission_rate(self,
            new_ensemble_transmission_rate,
            old_ensemble_transmission_rate,
            n_user_nodes,
            n_obs_nodes):
        new_ensemble_transmission_rate = ( \
                old_ensemble_transmission_rate*(n_user_nodes-n_obs_nodes) \
                                        + new_ensemble_transmission_rate*n_obs_nodes) \
                                        / n_user_nodes
        return new_ensemble_transmission_rate
