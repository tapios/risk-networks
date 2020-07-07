import numpy as np
import scipy.linalg as la
import time
from sklearn.decomposition import TruncatedSVD
 
class EnsembleAdjustmentKalmanFilter:

    def __init__(
            self,
            prior_svd_reduced = True,
            observation_svd_reduced = False,
            joint_cov_noise = 1e-2):
        '''
        Instantiate an object that implements an Ensemble Adjustment Kalman Filter.

        Key functions:
            * eakf.obs
            * eakf.update
            * eakf.compute_error
        Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        '''
        if not prior_svd_reduced and observation_svd_reduced:
            raise NotImplementedError("observation SVD cannot be reduced, if prior SVD is not reduced")
        
        # Error
        self.error = np.empty(0)
        self.prior_svd_reduced = prior_svd_reduced
        self.observation_svd_reduced = observation_svd_reduced
        self.joint_cov_noise = joint_cov_noise
     
        # Compute error
    def compute_error(
            self,
            x,
            x_t,
            cov):
        diff = x_t - x.mean(0)
        error = diff.dot(np.linalg.solve(cov, diff))
        # Normalize error
        norm = x_t.dot(np.linalg.solve(cov, x_t))
        error = error/norm

        self.error = np.append(self.error, error)


    # x: forward evaluation of state, i.e. x(q), with shape (num_ensembles, num_elements)
    # q: model parameters, with shape (num_ensembles, num_elements)
    def update(
            self,
            ensemble_state,
            clinical_statistics,
            transmission_rates,
            truth,
            cov,
            print_error=False,
            r=1.0):

        '''
        - ensemble_state (np.array): J x M of observed states for each of the J ensembles

        - clinical_statistics (np.array): transition rate model parameters for each of the J ensembles

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

        cov = np.clip((1./np.maximum(x_t, 1e-12)/np.maximum(1-x_t, 1e-12)), -5, 5)**2 * cov
        x_t = np.log(np.maximum(x_t, 1e-12)/np.maximum(1.-x_t, 1e-12))

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.linalg.LinAlgError:
            print('cov not invertible')
            cov_inv = np.ones(cov.shape)

        # States
        x = np.log(np.maximum(ensemble_state, 1e-9) / np.maximum(1.0 - ensemble_state, 1e-9))
        # Stacked parameters and states
        # the transition and transmission parameters act similarly in the algorithm
        p = clinical_statistics
        q = transmission_rates

        #if only 1 state is given
        if (ensemble_state.ndim == 1):
            x=x[np.newaxis].T

        if p.size>0 and q.size>0:
            zp = np.hstack([p, q, x])
        elif p.size>0 and q.size==0:
            zp = np.hstack([p,x])
        elif q.size>0 and p.size==0:
            zp = np.hstack([q, x])
        else:
            zp = x

        # Ensemble size
        J = x.shape[0]

        # Sizes of q and x
        pqs = q[0].size +p[0].size
        xs = x[0].size

        zp_bar = np.mean(zp, 0)

        H = np.hstack([np.zeros((xs, pqs)), np.eye(xs)])
        Hpq = np.hstack([np.eye(pqs), np.zeros((pqs, xs))])
        
        # Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        if not self.prior_svd_reduced:
            # Calculate the full prior covariance from zp
            Sigma = np.cov(zp.T)

            #if only one state is given
            if Sigma.ndim < 2:
                Sigma=np.array([Sigma])
                Sigma=Sigma[np.newaxis]
                
            # Add noise directly to the sample covariance for ill-conditioned Sigma
            Sigma = Sigma + np.diag(np.full(pqs+xs, self.joint_cov_noise))

            # Perform SVD on Sigma
            F, Dp_vec, _ = la.svd(Sigma)
            F_full = F
            Dp_vec_full = Dp_vec
            Dp = np.diag(Dp_vec_full)

        else:

            # if ensemble_size < observations size, we pad the singular value matrix with added noise
            if zp.shape[0] < zp.shape[1]:
                F_full, Dp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
                F = F_full[:,:J-1]
                Dp_vec = Dp_vec[:-1]
                Dp_vec = 1./np.sqrt(J-1) * Dp_vec
                Dp_vec_full = np.zeros(zp.shape[1])
                Dp_vec_full[:J-1] = Dp_vec
                Dp_vec_full = Dp_vec_full**2 + self.joint_cov_noise 
                Dp = np.diag(Dp_vec_full)
                Sigma = np.linalg.multi_dot([F_full, Dp, F_full.T])
            
            else:
                F_full, Dp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
                F = F_full
                Dp_vec = 1./np.sqrt(J-1) * Dp_vec
                Dp_vec_full = Dp_vec**2 + self.joint_cov_noise 
                Dp = np.diag(Dp_vec_full)
                Sigma = np.linalg.multi_dot([F_full, Dp, F_full.T])


        # NB: prior_svd_reduced == False implies observation_svd_reduced == False         
        if not self.observation_svd_reduced:
            start = time.perf_counter()

            # Performing the second SVD of EAKF in the full space
            G = np.diag(np.sqrt(Dp_vec_full))
            G_inv = np.diag(1./np.sqrt(Dp_vec_full))
            U, D_vec, _ = la.svd(np.linalg.multi_dot([G.T, F_full.T, H.T, np.sqrt(cov_inv)]), \
                                 full_matrices=True)
            D_vec = D_vec**2
            B = np.diag((1.0 + D_vec) ** (-1.0 / 2.0))
            A = np.linalg.multi_dot([F_full, \
                                     G.T, \
                                     U, \
                                     B.T, \
                                     G_inv, \
                                     F_full.T])
            Sigma_u = np.linalg.multi_dot([A, Sigma, A.T])
               
            end = time.perf_counter()
            print("Time for second SVD: ", end-start)

        else:

            start = time.perf_counter()

            Dp_vec = Dp_vec**2 + self.joint_cov_noise 
            G = np.diag(np.sqrt(Dp_vec))
            G_inv = np.diag(1./np.sqrt(Dp_vec))

            # first obtain the singular vectors
            U, _, _ = la.svd(np.linalg.multi_dot([G.T, F.T, H.T, np.sqrt(cov_inv)]),)

            #
            trunc_size = min(J-1,cov_inv.shape[0])
            trunc_svd = TruncatedSVD(n_components=trunc_size, random_state=42)
            trunc_svd.fit(np.linalg.multi_dot([G.T, F_full.T, H.T, np.sqrt(cov_inv)]))
            sing_val = trunc_svd.singular_values_
            min_sing_val_sq = np.min(sing_val)**2
            D_vec = min_sing_val * np.ones(F_full.shape[0])
            D_vec[:trunc_size] = sing_val*sing_val  
            B = np.diag((1.0 + D_vec) ** (-1.0 / 2.0))
            A = np.linalg.multi_dot([F, \
                                     G.T, \
                                     U, \
                                     B.T, \
                                     G_inv, \
                                     F.T])
            
            Sigma = np.linalg.multi_dot([F_full, Dp, F_full.T])
            Sigma_u = np.linalg.multi_dot([A, Sigma, A.T])
            
            # ## previous implementation
            # U, D_vec, _ = la.svd(np.linalg.multi_dot([G.T, F.T, H.T, np.sqrt(cov_inv)]))
            # D_vec = D_vec**2
            # B = np.diag((1.0 + D_vec) ** (-1.0 / 2.0))
            # A = np.linalg.multi_dot([F, \
            #                          G.T, \
            #                          U, \
            #                          B.T, \
            #                          G_inv, \
            #                          F.T])
            
            # Sigma = np.linalg.multi_dot([F_full, Dp, F_full.T])
            # Sigma_u = np.linalg.multi_dot([A, Sigma, A.T])
               

            # if zp.shape[0] < zp.shape[1]:
            #     ## Adding noises to Sigma_u
            #     zu_tmp = np.dot(A,  1./np.sqrt(J-1) * (zp-np.mean(zp,0)).T)
            #     F_u_full, _, _ = la.svd(zu_tmp, full_matrices=True)
               
            #     svd1 = TruncatedSVD(n_components=J-1, random_state=42)
            #     svd1.fit(Sigma_u)
            #     Dp_u_vec = svd1.singular_values_
            #     Dp_u_vec_full = np.ones(F_u_full.shape[0]) * Dp_u_vec[-1]
            #     Dp_u_vec_full[:J-1] = Dp_u_vec
            #     Dp_u = np.diag(Dp_u_vec_full)
            #     Sigma_u = np.linalg.multi_dot([F_u_full, Dp_u, F_u_full.T])

            end = time.perf_counter()
            print("Time for second SVD: ", end-start)

        Sigma_inv = np.linalg.multi_dot([F_full, np.linalg.inv(Dp), F_full.T])
        zu_bar = np.dot(Sigma_u, \
                           (np.dot(Sigma_inv, zp_bar) + np.linalg.multi_dot([H.T, cov_inv, x_t])))

        # Update parameters and state in `zu`
        zu = np.dot(zp - zp_bar, A.T) + zu_bar

        # Store updated parameters and states
        x_logit = np.dot(zu, H.T)

        # Avoid overflow for exp
        x_logit = np.minimum(x_logit, 1e2)

        # replace unchanged states
        new_ensemble_state = np.exp(x_logit)/(np.exp(x_logit) + 1.0)

        pqout=np.dot(zu,Hpq.T)
        new_clinical_statistics = pqout[:, :clinical_statistics.shape[1]]
        new_transmission_rates  = pqout[:, clinical_statistics.shape[1]:]
        #self.x = np.append(self.x, [x_p], axis=0)

        if (ensemble_state.ndim == 1):
            new_ensemble_state=new_ensemble_state.squeeze()

        # Compute error
        if print_error:
            self.compute_error(x_logit,x_t,cov)

        #print("new_clinical_statistics", new_clinical_statistics)
        #print("new_transmission_rates", new_transmission_rates)
        return new_ensemble_state, new_clinical_statistics, new_transmission_rates
