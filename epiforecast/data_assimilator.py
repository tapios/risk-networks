import numpy as np
import copy


from epiforecast.ensemble_adjustment_kalman_filter import EnsembleAdjustmentKalmanFilter

class DataAssimilator:

    def __init__(
            self,
            observations,
            errors,
            *,
            n_assimilation_batches=1,
            transition_rates_to_update_str=None,
            transmission_rate_to_update_flag=None):
        """
           A data assimilator, to perform updates of model parameters and states using an
           ensemble adjustment Kalman filter (EAKF) method.

           Positional Args
           ---------------
           observations (list, [], or Observation): A list of Observations, or a single Observation.
                                                     Generates the indices and covariances of observations

           errors (list, [],  or Observation): Observation for the purpose of error checking. Error
                                                observations are used to compute online differences at
                                                the observed (according to Errors) between Kinetic and
                                                Master Equation models
           Keyword Args
           ------------
           n_assimilation_batches (int) : Default = 1, the number of random batches over which we assimilate.
                                          At the cost of information loss, one can batch assimilation updates into random even-sized batches,
                                          the update scales with O(num observation states^3) and is memory intensive.
                                          Thus performing n x m-sized updates is far cheaper than an nm-sized update. 
         
                                          
           transition_rates_to_update_str (list): list of strings naming the transition_rates we would like
                                                  update with data. must coincide with naming found in
                                                  epiforecast/populations.py.
                                                  If not provided, will set []

           transmission_rate_to_update_flag (boolean): bool to update transmission rate with data
                                                       If not provided will set False
           Methods
           -------

           update(ensemble_state, data, contact_network=[], full_ensemble_transition_rates, full_ensemble_transmission_rate):
               Perform an update of the ensemble states `ensemble_state`, and if provided, ensemble
               parameters `full_ensemble_transition_rates`, `full_ensemble_transmission_rate` and the network
               `contact network`. Returns the updated model parameters and updated states.

           make_new_observation(state): For every Observation model, update the list of indices at which to observe (given by observations.obs_states).
                                        Returns a concatenated list of indices `observed_states` with duplicates removed.

           get_observation_cov(): For every Observation model, obtain the relevant variances when taking a measurement of data. Note we account for multiple node
                                  measurements by using the minimum variance at that node (I.e if same node is queried twice in a time interval we take the most
                                  accurate test). Returns a diagonal covariance matrix for the distinct observed states, with the minimum variances on the diagonal.

           sum_to_one(state): Takes the state `state` and enforces that all statuses at a node sum to one. Does this by distributing the mass (1-(I+H+R+D)) into S and E, where
                              the mass is divided based on the previous state's relative mass in S and E. i.e Snew= S/(S+E)*(1-(I+H+R+D)), Enew = E/(S+E)*(1-(I+H+R+D))

           error_to_truth_state(state,data): updates emodel.obs_states and measures (user prescribed) differences between the data and state online.
                                             #Current implementation sums the difference in number of predicted states
                                             #and actual states in an given interval e.g 0.5 <= I <= 1.0
        """

        if not isinstance(observations, list): # if it's a scalar, not array
            observations = [observations]

        if not isinstance(errors, list): # if it's a scalar, not array
            observations = [errors]

        if transition_rates_to_update_str is None:
            transition_rates_to_update_str = []

        if transmission_rate_to_update_flag is None:
            transmission_rate_to_update_flag = False

        if not isinstance(transition_rates_to_update_str,list):#if it's a string, not array
            transition_rates_to_update_str = [transition_rates_to_update_str]

        # observation models(s)
        self.observations = observations

        # the data assimilation models (One for each observation model)
        self.damethod = EnsembleAdjustmentKalmanFilter(prior_svd_reduced=True, \
                                                       observation_svd_reduced=False)
        # number of batches to assimilate data ('localization')
        self.n_assimilation_batches=n_assimilation_batches
        
        # online evaluations of errors, one needs an observation class to check differences in data
        self.online_emodel= errors

        # which parameter to assimilate joint with the state
        self.transition_rates_to_update_str = transition_rates_to_update_str
        self.transmission_rate_to_update_flag = transmission_rate_to_update_flag

        # storage for observations time : obj 
        self.stored_observed_states = {}
        self.stored_observed_nodes = {}
        self.stored_observed_means = {}
        self.stored_observed_variances = {}
 

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
            observed_nodes = self.stored_observed_nodes[current_time]
            return observed_states, observed_nodes
        
        else:
            observed_states = []
            for observation in self.observations:
                observation.find_observation_states(user_network,
                                                    ensemble_state,
                                                    data)
                if observation.obs_states.size > 0:
                    observed_states.extend(observation.obs_states)
                    if verbose:
                        print("[ Data assimilator ]",
                              observation.name,
                              ":",
                              len(observation.obs_states))

            observed_states = np.array(observed_states)
            #observed_nodes = np.unique(np.remainder(observed_states,self.observations[0].N))
            observed_nodes_raw = np.remainder(observed_states,self.observations[0].N)
            obs_num = observed_nodes_raw.shape[0]
            if np.array_equal(observed_nodes_raw[:int(obs_num/2)], observed_nodes_raw[int(obs_num/2):]):
                observed_nodes = observed_nodes_raw[:int(obs_num/2)]
            else:
                observed_nodes = observed_nodes_raw
            self.stored_observed_states[current_time] = observed_states
            self.stored_observed_nodes[current_time] = observed_nodes
            
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
            current_time,
            user_network=None,
            verbose=False,
            print_error=False,
            global_update=False):
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
                print("[ Data assimilator ] Total states to be assimilated: ",
                      obs_states.size)

                # extract only those rates which we wish to update with DA
                (ensemble_transition_rates,
                 ensemble_transmission_rate
                ) = self.extract_model_parameters_to_update(
                        full_ensemble_transition_rates,
                        full_ensemble_transmission_rate,
                        obs_nodes)
               
                print("[ Data assimilator ] Total parameters to be assimilated: ",np.hstack([ensemble_transition_rates,ensemble_transmission_rate]).shape[1])

                # Load the truth, variances of the observation(s)
                truth = self.stored_observed_means[current_time]
                var = self.stored_observed_variances[current_time]
                cov = np.diag(var)

                # Perform da model update with ensemble_state: states, transition and transmission rates 
                prev_ensemble_state = copy.deepcopy(ensemble_state)

                if global_update == True:
                    # If batching is not required:
                    if self.n_assimilation_batches==1:
                        if user_network == None:
                            total_nodes_num = int(ensemble_state.shape[1]/5)
                            update_states_index_min = int(np.round(np.min(obs_states/total_nodes_num)) \
                                                          * total_nodes_num)
                            update_states_index_max = int(np.ceil(np.max(obs_states/total_nodes_num)) \
                                                          * total_nodes_num)
                            update_states_num = update_states_index_max - update_states_index_min
                            H_obs = np.zeros([obs_nodes.size, update_states_num])
                            H_obs[list(range(obs_nodes.size)),obs_nodes] = 1

                            (ensemble_state[:, update_states_index_min:update_states_index_max],
                             new_ensemble_transition_rates,
                             new_ensemble_transmission_rate
                            ) = self.damethod.update(ensemble_state[:, update_states_index_min:update_states_index_max],
                                                     ensemble_transition_rates,
                                                     ensemble_transmission_rate,
                                                     truth,
                                                     cov,
                                                     H_obs,
                                                     print_error=print_error)
                        else:
                            neighbour_nodes = user_network.get_neighbors(obs_nodes)
                            update_states_nodes = np.hstack([neighbour_nodes,obs_nodes]) 
                            update_states_num = update_states_nodes.size
                            H_obs = np.hstack([np.zeros((obs_nodes.size,neighbour_nodes.size)), 
                                               np.eye(obs_nodes.size)])

                            (ensemble_state[:, update_states_nodes],
                             new_ensemble_transition_rates,
                             new_ensemble_transmission_rate
                            ) = self.damethod.update(ensemble_state[:, update_states_nodes],
                                                     ensemble_transition_rates,
                                                     ensemble_transmission_rate,
                                                     truth,
                                                     cov,
                                                     H_obs,
                                                     print_error=print_error)

                else:
                    # If batching is not required:
                    if self.n_assimilation_batches==1:
                        H_obs = np.eye(obs_states.size)
                        (ensemble_state[:, obs_states],
                         new_ensemble_transition_rates,
                         new_ensemble_transmission_rate
                        ) = self.damethod.update(ensemble_state[:, obs_states],
                                                 ensemble_transition_rates,
                                                 ensemble_transmission_rate,
                                                 truth,
                                                 cov,
                                                 H_obs,
                                                 print_error=print_error)
                    else: #perform the EAKF in batches
                        
                        #create batches, with final batch larger due to rounding
                        batch_size = int(obs_states.size / self.n_assimilation_batches)
                        permuted_idx = np.random.permutation(np.arange(obs_states.size))
                        batches =[ permuted_idx[i * batch_size:(i + 1) * batch_size] if i < (self.n_assimilation_batches - 1)
                                   else permuted_idx[i * batch_size:]
                                   for i in np.arange(self.n_assimilation_batches)]
                        
                        for batch in batches:
                            H_obs = np.eye(batch.size)
                            cov_batch = np.diag(np.diag(cov)[batch])
                            (ensemble_state[:, obs_states[batch]],
                             new_ensemble_transition_rates,
                             new_ensemble_transmission_rate
                            ) = self.damethod.update(ensemble_state[:, obs_states[batch]],
                                                     ensemble_transition_rates,
                                                     ensemble_transmission_rate,
                                                     truth[batch],
                                                     cov_batch,
                                                     H_obs,
                                                     print_error=print_error)
                        
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


