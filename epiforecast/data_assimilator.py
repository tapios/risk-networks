import numpy as np
import copy

from epiforecast.ensemble_adjusted_kalman_filter import EnsembleAdjustedKalmanFilter

class DataAssimilator:

    def __init__(
            self,
            observations,
            errors,
            *,
            transition_rates_to_update_str=None,
            transmission_rate_to_update_flag=None):
        """
           A data assimilator, to perform updates of model parameters and states using an
           ensemble adjusted Kalman filter (EAKF) method.

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
        self.damethod = EnsembleAdjustedKalmanFilter()

        # online evaluations of errors, one needs an observation class to check differences in data
        self.online_emodel= errors

        # which parameter to assimilate joint with the state
        self.transition_rates_to_update_str = transition_rates_to_update_str
        self.transmission_rate_to_update_flag = transmission_rate_to_update_flag

    def find_observation_states(
            self,
            contact_network,
            ensemble_state,
            data):
        """
        Make all the observations in the list self.observations.

        This sets observation.obs_states.
        """
        print("Observation type : Number of Observed states")
        observed_states = []
        for observation in self.observations:
            observation.find_observation_states(contact_network, ensemble_state, data)
            print(observation.name,":",len(observation.obs_states))
            if observation.obs_states.size > 0:
                observed_states.extend(observation.obs_states)

       # observed_states = np.hstack([observation.obs_states for observation in self.observations])
        return np.array(observed_states)

    def observe(
            self,
            contact_network,
            state,
            data,
            scale='log',
            noisy_measurement=False):

        observed_means = []
        observed_variances = []
        for observation in self.observations:
            if (observation.obs_states.size >0):
                observation.observe(contact_network,
                                    state,
                                    data,
                                    scale,
                                    noisy_measurement)

                observed_means.extend(observation.mean)
                observed_variances.extend(observation.variance)

        #observed_means = np.hstack([observation.mean for observation in self.observations])
        #observed_variances= np.hstack([observation.variance for observation in self.observations])

        return np.array(observed_means), np.array(observed_variances)


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
            user_network):

        ensemble_size = ensemble_state.shape[0]

        if len(self.observations) == 0: # no update is performed; return input
            return ensemble_state, full_ensemble_transition_rates, full_ensemble_transmission_rate

        else:

            # Extract the transition_rates to update
            if len(self.transition_rates_to_update_str) > 0:

                ensemble_transition_rates = np.array([])

                for member in range(ensemble_size):

                    # This returns an [ensemble size x transition rates (to be updated)] np.array
                    rates_tmp = np.hstack([getattr(full_ensemble_transition_rates[member], rate_type)
                                           for rate_type in self.transition_rates_to_update_str])

                    # Have to create here as rates_tmp unknown in advance
                    if member == 0: ensemble_transition_rates = np.empty((0, rates_tmp.size), dtype=float)

                    ensemble_transition_rates = np.append(ensemble_transition_rates, [rates_tmp], axis=0)

                ensemble_transition_rates = np.vstack(ensemble_transition_rates)

            else: # set to column of empties
                ensemble_transition_rates =  np.empty((ensemble_size, 0), dtype=float)

            if self.transmission_rate_to_update_flag is True:
                ensemble_transmission_rate = full_ensemble_transmission_rate

            else: # set to column of empties
                ensemble_transmission_rate = np.empty((ensemble_size, 0), dtype=float)

            om = self.observations
            dam = self.damethod

            obs_states = self.find_observation_states(user_network, ensemble_state, data) # Generate states to observe
            if (obs_states.size > 0):

                print("Total states to be assimilated: ", obs_states.size)
                # Get the truth indices, for the observation(s)

                truth,var = self.observe(user_network,
                                         ensemble_state,
                                         data,
                                         scale = None)
                cov = np.diag(var)
                
                # Get the covariances for the observation(s), with the minimum returned if two overlap
                #cov = self.get_observation_cov()
                # Perform da model update with ensemble_state: states, transition and transmission rates
                
                prev_ensemble_state = copy.deepcopy(ensemble_state)
                (ensemble_state[:, obs_states],
                 new_ensemble_transition_rates,
                 new_ensemble_transmission_rate) = dam.update(ensemble_state[:, obs_states],
                                                              ensemble_transition_rates,
                                                              ensemble_transmission_rate,
                                                              truth,
                                                              cov)


                # print states > 1
                #tmp = ensemble_state.reshape(ensemble_state.shape[0],5,om[0].N)
                #sum_states = np.sum(tmp,axis=1)
                # print(sum_states[sum_states > 1 + 1e-2])

             
                self.sum_to_one(prev_ensemble_state, ensemble_state)
                   
                # print same states after the sum_to_one()
                # tmp = ensemble_state.reshape(ensemble_state.shape[0],5,om[0].N)
                # sum_states_after = np.sum(tmp,axis=1)
                # print(sum_states_after[sum_states > 1 + 1e-2]) #see what the sum_to_one did to them


                # Update the new transition rates if required
                if len(self.transition_rates_to_update_str) > 0:

                    for member in range(ensemble_size):

                        new_member_rates = new_ensemble_transition_rates[member,:]

                        for rate_type in self.transition_rates_to_update_str:
                            # Need to go back from numpy array to setting rates
                            # We obtain the size, then update the corresponding transition rate
                            # Then delete this an move onto the next rate
                            clinical_parameter = getattr(full_ensemble_transition_rates[member], rate_type)
                            if not isinstance(clinical_parameter, np.ndarray):
                                rate_size = np.array(clinical_parameter).size
                                new_rates=new_member_rates[0]
                            else:
                                rate_size=clinical_parameter.size
                                new_rates=new_member_rates[:rate_size]
                            full_ensemble_transition_rates[member].set_clinical_parameter(rate_type, new_rates)

                            new_member_rates = np.delete(new_member_rates, np.arange(rate_size))

                # Update the transmission_rate if required
                if self.transmission_rate_to_update_flag is True:
                    full_ensemble_transmission_rate=new_ensemble_transmission_rate


                print("EAKF error:", dam.error[-1])
            else:
                print("No assimilation required")

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

        #print(truth[actual_infected])

        different_states=np.zeros(em.n_status*em.N)
        different_states[predicted_infected]=1.0
        different_states[actual_infected] = different_states[actual_infected]-1.0

        number_different=np.maximum(different_states,0.0)-np.minimum(different_states,0.0)

        print("Differences between predicted and true I>0.5:",np.sum(number_different).astype(int))


    #as we measure a subset of states, we may need to enforce other states to sum to one

    def sum_to_one(
            self,
            prev_ensemble_state,
            ensemble_state):

        N=self.observations[0].N
        n_status=self.observations[0].n_status
        if n_status == 6:
            #First enforce probabilities == 1, by placing excess in susceptible and Exposed
            #split based on their current proportionality.
            #(Put all in S or E leads quickly to [0,1] bounding issues.
            tmp = ensemble_state.reshape(ensemble_state.shape[0],n_status,N)
            IHRDmass = np.sum(tmp[:,2:,:],axis=1) #sum over I H R D
            Smass = ensemble_state[:,0:N]#mass in S
            Emass = ensemble_state[:,N:2*N]#mass in E
            fracS = Smass/(Smass+Emass)#get the proportion of mass in frac1
            fracE = 1.0-fracS
            ensemble_state[:,0:N] = (1.0 - IHRDmass)*fracS #mult rows by fracS
            ensemble_state[:,N:2*N] =  (1.0 - IHRDmass)*fracE

        elif n_status==5:
            # First obtain the mass contained in category "E"
            prev_tmp = prev_ensemble_state.reshape(prev_ensemble_state.shape[0],n_status, N)
            Emass = 1.0 - np.sum(prev_tmp,axis=1) # E= 1 - (S + I + H + R + D)
            # for each observation we get the observed status e.g 'I' and fix it (as it was updated)
            # we then normalize the other states e.g (S,'E',H,R,D) over the difference 1-I
            for observation in self.observations:
                if len(observation.obs_states > 0):
                   
                    observed_nodes = np.remainder(observation.obs_states,N)
                    updated_status = observation.obs_status_idx
                
                    free_statuses = [ i for i in range(5) if i!= updated_status]
                    tmp = ensemble_state.reshape(ensemble_state.shape[0],n_status, N)
               
                    # create arrays of the mass in the observed and the unobserved "free" statuses at the observed nodes.
                    observed_tmp = tmp[:,:,observed_nodes]
                    updated_mass = observed_tmp[:, updated_status, :]
                    free_states  = observed_tmp
                    free_states[:, updated_status, :]  = np.zeros([free_states.shape[0], 1, free_states.shape[2]]) #remove this axis for the sum (but maintain the shape)
                  
                    free_mass = np.sum(free_states,axis=1) + Emass[:,observed_nodes]
                    
                    # normalize the free values e.g for S: set S = (1-I) * S/(S+E+H+R+D)
                    for i in free_statuses:
                        
                        ensemble_state[:, i*N+observed_nodes] = (1.0 - updated_mass[:,0,:]) * (free_states[:, i, :] / free_mass)
                    
