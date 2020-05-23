import numpy as np
import scipy.linalg as la

class EnsembleAdjustedKalmanFilter:

    def __init__(self, params_cov_noise = 1e-2, states_cov_noise = 1e-2, params_noise_active = False, states_noise_active = True):
        '''
        Instantiate an object that implements an Ensemble Adjusted Kalman Filter.

        Key functions:
            * eakf.obs
            * eakf.update
            * eakf.compute_error
        Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        '''
        
        # Error
        self.error = np.empty(0)
        self.params_cov_noise = params_cov_noise
        self.states_cov_noise = states_cov_noise
        self.params_noise_active = params_noise_active
        self.states_noise_active = states_noise_active

        # Compute error
    def compute_error(self, x, x_t, cov):
        diff = x_t - x.mean(0)
        error = diff.dot(np.linalg.solve(cov, diff))
        # Normalize error
        norm = x_t.dot(np.linalg.solve(cov, x_t))
        error = error/norm

        self.error = np.append(self.error, error)

   
    # x: forward evaluation of state, i.e. x(q), with shape (num_ensembles, num_elements)
    # q: model parameters, with shape (num_ensembles, num_elements)
    def update(self, ensemble_state, transition_rates,transmission_rates, truth, cov, r=1.0):

        '''
        - ensemble_state (np.array): J x M of observed states for each of the J ensembles
        
        - transition_rates (np.array): transition rate model parameters for each of the J ensembles

        - transmission_rates (np.array): transmission rate of model parameters for each of the J ensembles
        
        - truth (np.array): M x 1 array of observed states.

        - cov (np.array): M x M array of covariances that represent observational uncertainty.
                          For example, an array of 0's represents perfect certainty.
                          Off-diagonal elements represent the fact that observations of state
                          i may not be independent from observations of state j. For example, this
                          can occur when a test applied to person ni alters the certainty of a subsequent
                          test to person nj.

        #TODO: how to deal with no transition and/or transmission rates. i.e empty array input.
               (Could we just use an ensemble sized column of zeros? then output the empty array
                '''

        assert (truth.ndim == 1), 'EAKF init: truth must be 1d array'
        assert (cov.ndim == 2), 'EAKF init: covariance must be 2d array'
        assert (truth.size == cov.shape[0] and truth.size == cov.shape[1]),\
            'EAKF init: truth and cov are not the correct sizes'
        
        # Observation data statistics at the observed nodes
        x_t = truth
        cov = r**2 * cov

        cov = (1./np.maximum(x_t, 1e-9)/np.maximum(1-x_t, 1e-9))**2 * cov
        x_t = np.log(np.maximum(x_t, 1e-9)/np.maximum(1.-x_t, 1e-9))

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.linalg.LinAlgError:
            print('cov not invertible')
            cov_inv = np.ones(cov.shape)

        # States
        x = np.log(np.maximum(ensemble_state, 1e-9) / np.maximum(1.0 - ensemble_state, 1e-9))

        # Stacked parameters and states
        # the transition and transmission parameters act similarly in the algorithm
        p=transition_rates
        q=transmission_rates
        
        zp = np.hstack([p, q, x])

        x_t = x_t
        cov = cov
        
        # Ensemble size
        J = x.shape[0]
        
        # Sizes of q and x
        pqs = q[0].size +p[0].size
        xs = x[0].size

        zp_bar = np.mean(zp, 0)
        Sigma = np.cov(zp.T)
        
        # Add noises to the diagonal of sample covariance 
        # Current implementation involves a small constant 
        # This numerical trick can be deactivated if Sigma is not ill-conditioned
        if self.params_noise_active == True:
            Sigma[:pqs,:pqs] = Sigma[:pqs,:pqs] + np.identity(pqs) * self.params_cov_noise 
        if self.states_noise_active == True:
            Sigma[pqs:,pqs:] = Sigma[pqs:,pqs:] + np.identity(xs) * self.states_cov_noise

        # Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        # Preparing matrices for EAKF 
        H = np.hstack([np.zeros((xs, pqs)), np.eye(xs)])
        Hpq = np.hstack([np.eye(pqs), np.zeros((pqs, xs))])
        F, Dp_vec, _ = la.svd(Sigma)
        Dp = np.diag(Dp_vec)
        G = np.diag(np.sqrt(Dp_vec))
        U, D_vec, _ = la.svd(np.linalg.multi_dot([G.T, F.T, H.T, cov_inv, H, F, G]))
        B = np.diag((1.0 + D_vec) ** (-1.0 / 2.0)) 
        A = np.linalg.multi_dot([np.linalg.inv(F.T), \
                                 G.T, \
                                 np.linalg.inv(U.T), \
                                 B.T, \
                                 np.linalg.inv(G.T), \
                                 F.T])
        Sigma_u = np.linalg.multi_dot([A, Sigma, A.T])
        
        ## Adding noises into data for each ensemble member (Currently deactivated)
        # noise = np.array([np.random.multivariate_normal(np.zeros(xs), cov) for _ in range(J)])
        # x_t = x_t + noise
        zu_bar = np.dot(Sigma_u, \
                           (np.dot(np.linalg.inv(Sigma), zp_bar) + np.dot(np.dot(H.T, cov_inv), x_t)))
    
        # Update parameters and state in `zu`
        zu = np.dot(zp - zp_bar, A.T) + zu_bar 

        # Store updated parameters and states
        x_logit = np.dot(zu, H.T)

        # Avoid overflow for exp
        x_logit = np.minimum(x_logit, 1e2)

        # replace unchanged states
        new_ensemble_state = np.exp(x_logit)/(np.exp(x_logit) + 1.0)

        pqout=np.dot(zu,Hpq.T)
        new_transition_rates,new_transmission_rates=pqout[:,:transition_rates.shape[1]],pqout[:,transition_rates.shape[1]:]
        
        #self.x = np.append(self.x, [x_p], axis=0)

        # Compute error
        self.compute_error(x_logit,x_t,cov)

        return new_ensemble_state, new_transition_rates, new_transmission_rates
