import numpy as np

import copy
import scipy.linalg as la

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
            full_svd=True,
            joint_cov_noise=1e-2,
            obs_cov_noise=0,
            inflate_states=True,
            inflate_reg=1.0,
            inflate_I_only=True,
            scale=None,
            transition_rates_min=None,
            transition_rates_max=None,
            transmission_rate_min=None,
            transmission_rate_max=None,
            distance_threshold=2,
            output_path=None):
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
                    Four values are supported: 'full_global', 'global', 'local',
                    'neighbor'.

            full_svd (bool): whether to use full or reduced second SVD in EAKF

            joint_cov_noise (float): Tikhonov-regularization noise

            inflate_states: inflate states in EAKF updating if True

            x_logit_std_threshold: threshold of std for inflation (% of mean value)

            inflate_I_only: only inflate I if True

            scale (string): 'log' or None for whether we take meausurements in a logit scale
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
                joint_cov_noise=joint_cov_noise,
                obs_cov_noise=obs_cov_noise,
                inflate_states = inflate_states,
                    inflate_reg = inflate_reg,
                    output_path=output_path)
        else:
            raise NotImplementedError("The implemetation of reduced second SVD has been removed!")

        self.inflate_I_only = inflate_I_only

        self.scale = scale

        if distance_threshold !=1:
            raise  NotImplementedError("only have implementation for distance_threshold=1")

        self.distance_threshold = distance_threshold
        # storage for observations time : obj 
        self.stored_observed_states = {}
        self.stored_observed_nodes = {}
        self.stored_observed_means = {}
        self.stored_observed_variances = {}
        self.stored_nodes_nearby_observed_state = {}
        self.stored_dist_to_observed_state = {}
        # range of transition rates
        self.transition_rates_min = transition_rates_min
        self.transition_rates_max = transition_rates_max 

        # range of transmission rate
        self.transmission_rate_min = transmission_rate_min 
        self.transmission_rate_max = transmission_rate_max 

        self.counter = 0
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
            observed_states = np.array(observed_states_list)
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
            noisy_measurement=True,
            verbose=False):

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
                                        self.scale)

                    observed_means.extend(observation.mean)
                    observed_variances.extend(observation.variance)
                    
                    if verbose:
                        print("[ Data assimilator ]",
                              observation.name,
                              ":",
                              observation.mean[0],
                              ", ",
                              observation.variance[0])
                        
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
            noisy_measurement=True,
            verbose=False):

        self.find_observation_states(user_network,
                                     ensemble_state,
                                     data,
                                     current_time,
                                     verbose=verbose)

        self.observe(user_network,
                     ensemble_state,
                     data,
                     current_time,
                     noisy_measurement=noisy_measurement,
                     verbose=verbose)

    def compute_inflate_indices(
            self,
            user_network):

        n_user_nodes = user_network.get_node_count()
        if self.inflate_I_only == True:
            inflate_states = np.arange(n_user_nodes, 2*n_user_nodes) 
        else:
            inflate_states = np.arange(5*n_user_nodes) 

        return inflate_states
               

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

    def get_nodes_near_observed(
            self,
            user_network,
            ostate):
        """
        Given the user_network, use  neighbourhood method to find nodes nearby the observation node.
        Assign a distance value to each node.

        Inputs
        ------
        user_network (networkx graph): the static user network (note the weights evolve)
        ostate (Int): a state index in the range [0:5*user_network.get_node_count()]

        Outputs
        -------
        nearby_obs  (np.array): node numbers that are nearby the ostate-node
        nearby_dist (np.array): distance value assigned to the obs 

        """

        onode = np.remainder(ostate, user_network.get_node_count()).tolist()
        #get neighbous
        nearby_obs = [onode]
        nearby_dist = [1]
        current_dist = 1
        new_nodelist = nearby_obs
        while self.distance_threshold >= current_dist:
            current_nodelist = copy.deepcopy(new_nodelist)
            new_nodelist = []
            for node in current_nodelist:
                neighbors = [i for i in  user_network.get_graph().neighbors(node)]
                new_nodelist.extend(neighbors)
                nearby_obs.extend(neighbors)
                #new_dist = [0 for i in range(len(neighbors))] # only observed weight 1 all others 0
                new_dist = [1 for i in range(len(neighbors))] # all neighbours weight 1
                #new_dist = [2/((current_dist+1)**2) for i in range(len(neighbors))] #1/2 
                nearby_dist.extend(new_dist)
                
            current_dist +=1
            
        nearby_obs = np.array(nearby_obs).astype(int)
        nearby_dist = np.array(nearby_dist)

        return nearby_obs, nearby_dist
    

                                 
    def update_initial_from_series(
            self,
            full_ensemble_state_series,
            full_ensemble_transition_rates,
            full_ensemble_transmission_rate,
            user_network,
            verbose=False,
            print_error=False):
        """
        given a time-indexed series of ensemble states over some time window [a,b],
        perform an EAKF update at the initial time (a) using the (stored) 
        observations taken over the window [a,b]

        Inputs
        -----
        full_ensemble_state_series (dict of np.arrays): holds {time : ensemble_state}
        full_ensemble_transition_rates (np.array): parameters for the ensemble [ens_size x params]
        full_ensemble_transmission_rate (np.array): parameters for the ensemble [ens_size x params]
        
        Outputs
        ------
        full_ensemble_state_series - with earliest time entry updated from EAKF
        full_ensemble_transition_rates - updated from EAKF
        full_ensemble_transmission_rate - updated from EAKF
    
        """
        
        update_flag = True

        if len(self.observations) == 0: # no update is performed; return input
            update_flag=False
            return full_ensemble_state_series, full_ensemble_transition_rates, full_ensemble_transmission_rate, update_flag

        #get the observation times from the assimilator
        observation_window = [min(full_ensemble_state_series.keys()), max(full_ensemble_state_series.keys())]
        observation_times = [obs_time for obs_time in self.stored_observed_states.keys() 
                             if  observation_window[0] <= obs_time <=observation_window[1] ]
        observation_times.sort()
        n_observation_times = len(observation_times)
       
        if n_observation_times == 0: # no update is performed; return input
            if verbose:
                print("[ Data assimilator ] No assimilation within window")
            update_flag=False
            return full_ensemble_state_series, full_ensemble_transition_rates, full_ensemble_transmission_rate, update_flag

        initial_time = observation_times[0]     
        # extract from full_ensemble_state_series
        full_ensemble_state_at_obs = {obs_time : full_ensemble_state_series[obs_time] for obs_time in observation_times}

        # Load states to compare with data
        obs_states = np.concatenate([self.stored_observed_states[obs_time] for obs_time in observation_times])
        tmp = [(obs_time*np.ones(self.stored_observed_states[obs_time].size)).tolist() for obs_time in observation_times]
        obs_idx_at_time = sum(tmp, []) # this just flattens the list
        obs_nodes = np.concatenate([self.stored_observed_nodes[obs_time] for obs_time in observation_times])
        total_obs_states = sum([self.stored_observed_states[obs_time].size for obs_time in observation_times])
        if (total_obs_states == 0):
            if verbose:
                print("[ Data assimilator ] No assimilation required")
            update_flag=False
            return full_ensemble_state_series, full_ensemble_transition_rates, full_ensemble_transmission_rate, update_flag

        if verbose:
            print("[ Data assimilator ] Total states over window with assimilation data: ",
                  total_obs_states)

        # extract only those rates which we wish to update with DA (will not change over window)
        (ensemble_transition_rates,
         ensemble_transmission_rate
        ) = self.extract_model_parameters_to_update(
            full_ensemble_transition_rates,
            full_ensemble_transmission_rate,
            user_network.get_nodes())  

        # Load the truth, variances of the observation(s)
        truth = np.concatenate([self.stored_observed_means[obs_time] for obs_time in observation_times]) 
        var = np.concatenate([self.stored_observed_variances[obs_time] for obs_time in observation_times])
        if verbose:
            print("mean for positive data: ",[tt for tt in truth if tt > 0] )

        # Perform DA model update with ensemble_state: states, transition and transmission rates
        initial_ensemble_state_pre_update = copy.deepcopy(full_ensemble_state_at_obs[initial_time])
        
        n_user_nodes = user_network.get_node_count()

        # Numbered according to Docs
        #[1.] Loop over observation states
        update_nodes = []
        for (os_idx,ostate) in enumerate(obs_states):
            #if it is a new observed state, find it, o/w load it 
            # note we use indices here as keys
            if os_idx in self.stored_nodes_nearby_observed_state:
                nearby_obs = self.stored_nodes_nearby_observed_state[os_idx]
            else:
                nearby_obs, nearby_dist =  self.get_nodes_near_observed(user_network, ostate)
                self.stored_nodes_nearby_observed_state[os_idx] = nearby_obs 
                self.stored_dist_to_observed_state[os_idx] = nearby_dist
                
            update_nodes.extend(nearby_obs)

        #contains all nodes that are 'nearby' an observation
        update_nodes = list(set(update_nodes))
        # [2.] now we loop one by one and assimilate each node
        for unode in update_nodes:
            
            update_states = np.array([unode + n_user_nodes*i for i in range(5)])
            
            os_idx_nearby_unode = []
            dist_to_obs_from_unode = []
            
            #[3.] find which of the observations unode is nearby, and how far it is away
            for (os_idx,ostate) in enumerate(obs_states):
                if unode in self.stored_nodes_nearby_observed_state[os_idx]:
                    os_idx_nearby_unode.extend([os_idx])
                    unode_idx = np.where(self.stored_nodes_nearby_observed_state[os_idx] == unode)                    
                    dist_to_obs_from_unode.extend( self.stored_dist_to_observed_state[os_idx][unode_idx].tolist() )

            # now we can build the joint state from update_states and the observation
            os_idx_times = [obs_idx_at_time[idx] for idx in os_idx_nearby_unode]
            os_idx_nearby_unode = np.array(os_idx_nearby_unode)
            dist_to_obs_from_unode = np.array(dist_to_obs_from_unode)
            
            # and create the effective data
            unode_truth = truth[os_idx_nearby_unode]
            unode_var = var[os_idx_nearby_unode] 
            unode_effective_cov = np.diag(unode_var/dist_to_obs_from_unode) # weight by distance function (larger distance = higher variance)
            
            # [4.] now build the joint state 3 cases
            # (1) we need to include all of unode states
            # (2) we need observed states at initial time (nearby to unode)
            # (3) we need observed states in future times (note this could include being at unode)
            
            # (1)
            unode_ensemble_state = full_ensemble_state_at_obs[initial_time][:,update_states]  #ens_size x 5

            # (2)
            unode_observed_state = []            
            os_idx_at_initial_time = [idx for (ii,idx) in enumerate(os_idx_nearby_unode) if abs(os_idx_times[ii] - initial_time) < 1e-6] #collect only observations at the current time
            os_idx_at_initial_not_at_unode = [ idx for idx in os_idx_at_initial_time if idx not in update_states ]

            if len(os_idx_at_initial_not_at_unode)>0:
                #the states at the current time
                obs_states_at_initial_not_at_unode = np.array([obs_states[os_idx] for os_idx in os_idx_at_initial_not_at_unode]).astype(int) #NB obs_states np.array
                
                #the data at this state and time
                state_at_obs_tmp = full_ensemble_state_at_obs[initial_time]
                unode_observed_state_tmp = state_at_obs_tmp[:,obs_states_at_initial_not_at_unode]
                unode_observed_state.append(unode_observed_state_tmp)

            # (3)
            for (time_idx,obs_time) in enumerate(observation_times[1:]):
                os_idx_at_obs_time = [idx for (ii,idx) in enumerate(os_idx_nearby_unode) if abs(os_idx_times[ii] - obs_time) < 1e-6] #collect only observations at the current time
                if len(os_idx_at_obs_time)>0:
                    #the states at the current time
                    obs_states_at_obs_time = np.array([obs_states[os_idx] for os_idx in os_idx_at_obs_time]).astype(int) #NB obs_states np.array
                    
                    #the data at this state and time
                    state_at_obs_tmp = full_ensemble_state_at_obs[obs_time]
                    unode_observed_state_tmp = state_at_obs_tmp[:,obs_states_at_obs_time]
                    unode_observed_state.append(unode_observed_state_tmp)
            
            if len(unode_observed_state) > 0:
                unode_joint_state = np.concatenate([unode_ensemble_state, *unode_observed_state], axis=1)
            else:
                unode_joint_state = unode_ensemble_state

            
            # [5.] Define the observation operator. to match the ordering of the joint state
            n_total_obs = sum(1 for idx in os_idx_nearby_unode)
            # remove the one case that unode[initial_time] is observed (similar to joint state)
            n_obs_at_unode_initial = 0
            for (i,obs) in enumerate(self.stored_observed_states[initial_time]):
                if (i in update_states) and (i in os_idx_nearby_unode):
                    n_obs_at_unode_initial += 1


            n_obs_not_at_unode_initial = n_total_obs - n_obs_at_unode_initial
            H_obs = np.zeros((unode_truth.shape[0],5 + n_obs_not_at_unode_initial))
          
            # split this unode[initial_time] case  and other (any node at later time, or other node at initial time)
            nonzero_idx=[]
            for obs in self.stored_observed_states[initial_time]: 
                if obs == unode:
                    j = np.where(update_states == obs) # in [0,1,2,3,4]
                    i = np.where(os_idx_nearby_unode == obs)  # of data
                    nonzero_idx.append([i,j])


            if len(nonzero_idx)>0:
                for idx in nonzero_idx:
                    H_obs[idx[0],idx[1]] = 1
                    np.delete(os_idx_nearby_unode, idx[1])
                    np.delete(dist_to_obs_from_unode,idx[1])
                    np.delete(os_idx_times,idx[1])


            #for future times, the ordering of truth & joint state mean that H_obs should be Identity
            for k in range(n_obs_not_at_unode_initial):
                H_obs[k,5+k] = 1
                
            #print(unode_joint_state.shape, H_obs.T.shape, unode_effective_cov.shape)
        
            #[6.] obtain which indices to inflate
            #inflate_indices = self.compute_inflate_indices(user_network)            
            inflate_indices = range(unode_joint_state.shape[1])

            #[7.] 
            (unode_joint_state,
             new_ensemble_transition_rates,
             new_ensemble_transmission_rate
            ) = self.damethod.update(unode_joint_state, #100 x 5*n_user_nodes + n_observation_times*n_observed_nodes
                                     ensemble_transition_rates, #100 x transi.
                                     ensemble_transmission_rate, #100 x transm.
                                     unode_truth, 
                                     unode_effective_cov, 
                                     H_obs, # 5*n_user_nodes + n_observation_times*n_observed_nodes                                     
                                     scale=self.scale,
                                     print_error=print_error,
                                     inflate_indices=inflate_indices,
                                     save_matrices=(self.counter == 0),
                                     save_matrices_name = str(observation_times[-1]))

            #print("joint state post DA", unode_joint_state.mean(axis=0))
            full_ensemble_state_series[initial_time][:,update_states] = unode_joint_state[:,:5]



            if self.transmission_rate_to_update_flag:
                # Clip transmission rate into a reasonable range
                new_ensemble_transmission_rate = np.clip(new_ensemble_transmission_rate,
                                                         self.transmission_rate_min,
                                                         self.transmission_rate_max)

                # Weighted-averaging based on ratio of observed nodes 
                new_ensemble_transmission_rate = self.weighted_averaged_transmission_rate(
                    new_ensemble_transmission_rate,
                    ensemble_transmission_rate,
                    n_user_nodes,
                    user_network.get_nodes()) # changed from particular observed nodes
                
        if len(self.transition_rates_to_update_str) > 0:
            new_ensemble_transition_rates = self.clip_transition_rates(new_ensemble_transition_rates,
                                                                           user_network.get_node_count())
                
        # set the updated rates in the TransitionRates object and
        # return the full rates.
        (full_ensemble_transition_rates,
         full_ensemble_transmission_rate
        ) = self.assign_updated_model_parameters(
            new_ensemble_transition_rates,
            new_ensemble_transmission_rate,
            full_ensemble_transition_rates,
            full_ensemble_transmission_rate,
            user_network.get_node_count())
        
        if print_error:
            print("[ Data assimilator ] EAKF error:", self.damethod.error[-1])

        # Error to truth
        if len(self.online_emodel)>0:
            self.error_to_truth_state(ensemble_state,data)

        # Return ensemble_state, transition rates, and transmission rate
        return full_ensemble_state_series, full_ensemble_transition_rates, full_ensemble_transmission_rate, update_flag
