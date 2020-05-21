import numpy as np
import pdb

class EAKF:

    # INPUTS:
    # parameters.shape = (num_ensembles, num_parameters)
    # states.shape = (num_ensembles, num_states)
    # Joint state: (q, x) but q only comes in update
    def __init__(self, states):
        '''
        Instantiate an object that implements an Ensemble Adjusted Kalman Filter.

        Key functions:
            * eakf.obs
            * eakf.update

        Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        '''

        assert (states.ndim == 2), \
            'EAKF init: states must be 2d array, num_ensembles x num_states'

        num_x = states.shape[1] 
        
        # Ensemble size
        self.J = states.shape[0]
        
        # States  
        #self.x = states[np.newaxis] 
  
        # Error
        self.error = np.empty(0)


        # Compute error
    def compute_error(self, x,x_t,cov):
        diff = x_t - x.mean(0)
        error = diff.dot(np.linalg.solve(cov, diff))
        # Normalize error
        norm = x_t.dot(np.linalg.solve(cov, x_t))
        error = error/norm

        self.error = np.append(self.error, error)

   
    # x: forward evaluation of state, i.e. x(q), with shape (num_ensembles, num_elements)
    # q: model parameters, with shape (num_ensembles, num_elements)
    def update(self, xin, q,truth,cov,r=1.0):

        '''
        - xin (np.array): J x M of observed states for each of the J ensembles
        
        - q (np.array): number of model parameters for each of the J ensembles
        
        - truth (np.array): M x 1 array of observed states.

        - cov (np.array): M x M array of covariances that represent observational uncertainty.
                          For example, an array of 0's represents perfect certainty.
                          Off-diagonal elements represent the fact that observations of state
                          i may not be independent from observations of state j. For example, this
                          can occur when a test applied to person ni alters the certainty of a subsequent
                          test to person nj.
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
        x = np.log(np.maximum(xin, 1e-9) / np.maximum(1.0 - xin, 1e-9))

        # Stacked parameters and states 
        
        zp = np.hstack([q, x])
        x_t = x_t
        cov = cov
        
        # Ensemble size
        J = self.J
        
        # Sizes of q and x
        qs = q[0].size
        xs = x[0].size
        
        # means and covariances
        q_bar = np.zeros(qs)
        x_bar = np.zeros(xs)
        c_qq = np.zeros((qs, qs))
        c_qx = np.zeros((qs, xs))
        c_xx = np.zeros((xs, xs))
        
        # Loop through ensemble to start computing means and covariances
        # (all the summations only)
        for j in range(J):
            
            q_hat = q[j]
            x_hat = x[j]
            
            # Means
            q_bar += q_hat
            x_bar += x_hat
            
            # Covariance matrices
            c_qq += np.tensordot(q_hat, q_hat, axes=0)
            c_qx += np.tensordot(q_hat, x_hat, axes=0)
            c_xx += np.tensordot(x_hat, x_hat, axes=0)
            
        # Finalize means and covariances
        # (divide by J, subtract of means from covariance sum terms)
        q_bar = q_bar / J
        x_bar = x_bar / J
        c_qq  = c_qq  / J - np.tensordot(q_bar, q_bar, axes=0)
        c_qx  = c_qx  / J - np.tensordot(q_bar, x_bar, axes=0)
        c_xx  = c_xx  / J - np.tensordot(x_bar, x_bar, axes=0)

        # Add noises to the diagonal of sample covariance 
        # Current implementation involves a small constant 
        # (1e-6 for original space, 1e-2 for logistic transformed space).
        # This numerical trick can be deactivated if Sigma is not ill-conditioned
        # c_xx = c_xx + np.identity(c_xx.shape[0]) * 1e-6 
        c_xx = c_xx + np.identity(c_xx.shape[0]) * 1e-2 

        # Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        # Preparing matrices for EAKF 
        Sigma  = np.vstack([np.hstack([c_qq, c_qx]), np.hstack([c_qx.T, c_xx])])
        H = np.hstack([np.zeros((xs, qs)), np.eye(xs)])
        Hq = np.hstack([np.eye(qs), np.zeros((qs, xs))])
        F, Dp_vec, _ = np.linalg.svd(Sigma)
        Dp = np.diag(Dp_vec)
        G = np.diag(np.sqrt(Dp_vec))
        U, D_vec, _ = np.linalg.svd(np.linalg.multi_dot([G.T, F.T, H.T, cov_inv, H, F, G]))
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
        zp_bar = np.hstack([q_bar, x_bar])
        zu_bar = np.matmul(Sigma_u, \
                           (np.dot(np.linalg.inv(Sigma), zp_bar) + np.dot(np.matmul(H.T, cov_inv), x_t)))
    
        # Update parameters and state in `zu`
        zu = np.dot(zp - zp_bar, A.T) + zu_bar 

        # Store updated parameters and states
        x_logit = np.dot(zu, H.T)

        # replace unchanged states
        x_p = np.exp(x_logit)/(np.exp(x_logit) + 1.0)

        qout=np.dot(zu,Hq.T)

        #self.x = np.append(self.x, [x_p], axis=0)

        # Compute error
        self.compute_error(x_logit,x_t,cov)

        return qout,x_p 
