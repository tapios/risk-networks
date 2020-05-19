import numpy as np
import scipy as sp
from sklearn.decomposition import TruncatedSVD

class EAKF:

    # INPUTS:
    # parameters.shape = (num_ensembles, num_parameters)
    # states.shape = (num_ensembles, num_states)
    # Joint state: (q, x)
    def __init__(self, parameters,states):
        '''
        Instantiate an object that implements an Ensemble Adjusted Kalman Filter.

        Key functions:
            * eakf.obs
            * eakf.update

        Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        '''

        assert (parameters.ndim == 2), \
            'EAKF init: parameters must be 2d array, num_ensembles x num_parameters'
        assert (states.ndim == 2), \
            'EAKF init: states must be 2d array, num_ensembles x num_states'

        num_x = states.shape[1] 
        num_q = parameters.shape[1]

        # Parameters
        self.q = parameters[np.newaxis]

        # Ensemble size
        self.J = parameters.shape[0]
        
        # States  
        self.x = states[np.newaxis] 
  
        # Error
        self.error = np.empty(0)


        # Compute error
    def compute_error(self, x):
        diff = self.x_t - x.mean(0)
        error = diff.dot(np.linalg.solve(self.cov, diff))
        # Normalize error
        norm = self.x_t.dot(np.linalg.solve(self.cov, self.x_t))
        error = error/norm

        self.error = np.append(self.error, error)

    # Take observation
    def obs(self, truth, cov, r = 1.0):
        '''
        Parameters
        ==========

        - truth (np.array): M x 1 array of observed states.
                            For a population of size N, there are M = 6*N states.

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
        self.x_t = truth
        self.cov = r**2 * cov

        self.cov = (1./np.maximum(self.x_t, 1e-9)/np.maximum(1-self.x_t, 1e-9))**2 * self.cov
        self.x_t = np.log(np.maximum(self.x_t, 1e-9)/np.maximum(1.-self.x_t, 1e-9))

        try:
            self.cov_inv = np.linalg.inv(cov)
        except np.linalg.linalg.LinAlgError:
            print('cov not invertible')
            self.cov_inv = np.ones(cov.shape)
       
    # x: forward evaluation of state, i.e. x(q), with shape (num_ensembles, num_elements)
    # q: model parameters, with shape (num_ensembles, num_elements)
    def update(self, xin):

        # States
        x = np.log(np.maximum(xin, 1e-9) / np.maximum(1.0 - xin, 1e-9))

        # Parameters
        q = np.copy(self.q[-1])

        # Stacked parameters and states 
        
        zp = np.hstack([q, x])
        x_t = self.x_t
        cov = self.cov
        
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
        c_qq = c_qq + np.identity(c_qq.shape[0]) * 1e0 

        # Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        # Preparing matrices for EAKF 
        Sigma  = np.vstack([np.hstack([c_qq, c_qx]), np.hstack([c_qx.T, c_xx])])
        H = np.hstack([np.zeros((xs, qs)), np.eye(xs)])
        Hq = np.hstack([np.eye(qs), np.zeros((qs, xs))])
        #svd1 = TruncatedSVD(n_components=Sigma.shape[0]-1, random_state=42)
        svd1 = TruncatedSVD(n_components=J, random_state=42)
        svd1.fit(Sigma)
        F = svd1.components_.T
        Dp_vec = svd1.singular_values_
        Dp = np.diag(Dp_vec)
        Sigma_inv = np.linalg.multi_dot([F, np.linalg.inv(Dp), F.T])
        G = np.diag(np.sqrt(Dp_vec))
        G_inv = np.diag(1./np.sqrt(Dp_vec))
        U, D_vec, _ = np.linalg.svd(np.linalg.multi_dot([G.T, F.T, H.T, self.cov_inv, H, F, G]))
        B = np.diag((1.0 + D_vec) ** (-1.0 / 2.0))
        A = np.linalg.multi_dot([F, \
                                 G.T, \
                                 U, \
                                 B.T, \
                                 G_inv, \
                                 F.T])
        Sigma_u = np.linalg.multi_dot([A, Sigma, A.T])

        # Adding noises (model uncertainties) back to the truncated dimensions
        # Need to further think about how to reduce the cost of full SVD here
        F_u, Dp_u_vec, _ = sp.linalg.svd(Sigma_u)
        Dp_u_vec[J:] = 1e-3
        Sigma_u = np.linalg.multi_dot([F_u, np.diag(Dp_u_vec), F_u.T])

        ## Adding noises into data for each ensemble member (Currently deactivated)
        # noise = np.array([np.random.multivariate_normal(np.zeros(xs), cov) for _ in range(J)])
        # x_t = x_t + noise
        zp_bar = np.hstack([q_bar, x_bar])
        zu_bar = np.matmul(Sigma_u, \
                           (np.dot(Sigma_inv, zp_bar) + np.dot(np.matmul(H.T, self.cov_inv), x_t)))    

        # Update parameters and state in `zu`
        zu = np.dot(zp - zp_bar, A.T) + zu_bar 

        # Store updated parameters and states
        x_logit = np.dot(zu, H.T)

        # Avoid overflow for exp
        x_logit = np.minimum(x_logit, 1e2)

        # replace unchanged states
        x_p = np.exp(x_logit)/(np.exp(x_logit) + 1.0)
       
        self.q = np.append(self.q, [np.dot(zu, Hq.T)], axis=0)
        self.x = np.append(self.x, [x_p], axis=0)

        # Compute error
        self.compute_error(x_logit)
