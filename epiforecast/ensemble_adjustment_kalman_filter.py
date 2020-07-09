import numpy as np
import scipy.linalg as la
import time
from sklearn.utils.extmath import randomized_svd
 
class EnsembleAdjustmentKalmanFilter:

    def __init__(
            self,
            prior_svd_reduced = True,
            observation_svd_reduced = True,
            observation_svd_regularized = False,
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
        self.observation_svd_regularized = observation_svd_regularized
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
                F_full, rtDp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
                F = F_full[:,:J-1]
                rtDp_vec = rtDp_vec[:-1]
                rtDp_vec = 1./np.sqrt(J-1) * rtDp_vec
                rtDp_vec_full = np.zeros(zp.shape[1])
                rtDp_vec_full[:J-1] = rtDp_vec
                Dp_vec_full = rtDp_vec_full**2 + self.joint_cov_noise 
                Dp = np.diag(Dp_vec_full)
            
            else:   
                F_full, rtDp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
                F = F_full
                rtDp_vec = 1./np.sqrt(J-1) * rtDp_vec
                Dp_vec_full = rtDp_vec**2 + self.joint_cov_noise 
                Dp = np.diag(Dp_vec_full)

            # compute np.linalg.multi_dot([F_full, Dp, F_full.T])
            Sigma = np.linalg.multi_dot([np.multiply(F_full, np.diag(Dp)),F_full.T])
            
        G_full = np.diag(np.sqrt(Dp_vec_full))
        G_inv_full = np.diag(1./np.sqrt(Dp_vec_full))
            
        # observation_svd_regularized == False utilizes the new implmentation of second svd
        # NB: prior_svd_reduced == False implies observation_svd_reduced == False         
        if not self.observation_svd_regularized:
            if not self.observation_svd_reduced:

                # Performing the second SVD of EAKF in the full space
                # This indent creates U and D_vec
               
                # computation of multidot([G_full.T, F_full.T, H.T, np.sqrt(cov_inv)])
                U, rtD_vec, _ = la.svd(np.linalg.multi_dot([np.multiply(F_full,np.diag(G_full)).T, np.multiply(H.T, np.sqrt(np.diag(cov_inv)))]), \
                                       full_matrices=True)
                D_vec = np.zeros(F_full.shape[0])
                D_vec[:cov_inv.shape[0]] = rtD_vec**2
            else:         
                # This indent creates U and D_vec       

                #get truncation size for the svd
                #(If too slow - one can try reducing 2*(J-1))
                trunc_size = min(2*(J-1),cov_inv.shape[0])

                #if cov_inv is small then no truncation required
                if trunc_size == cov_inv.shape[0]:
                    # computation of multidot([G_full.T, F_full.T, H.T, np.sqrt(cov_inv)])

                    U, rtD_vec, _ = la.svd(np.linalg.multi_dot([np.multiply(F_full,np.diag(G_full)).T, np.multiply(H.T, np.sqrt(np.diag(cov_inv)))]), \
                                           full_matrices=True)
                    D_vec = rtD_vec**2
                              
                else:
                    # The max number of singular values is the size of the observations
                    # we pad from trunc_size -> size obs, and pad from size_obs to joint space size

                    # calculating np.linalg.multi_dot([G_full.T, F_full.T, H.T, np.sqrt(cov_inv)]
                    Urect, rtD_vec , _ = randomized_svd(np.linalg.multi_dot([np.multiply(F_full,np.diag(G_full)).T,  np.multiply(H.T, np.sqrt(np.diag(cov_inv)))]), 
                                                        n_components=trunc_size,
                                                        power_iteration_normalizer = 'auto',
                                                        n_iter=10,
                                                        random_state=None)

                    # to get the full space, U, we pad it with a basis of the null space 
                    Unull = la.null_space(Urect.T)
                    U=np.hstack([Urect,Unull])
                      
                    # pad square rtD_vec and pad  with its smallest value, then with zeros
                    sing_val_sq = rtD_vec**2           
                    D_vec = np.hstack([sing_val_sq[-1] * np.ones(cov_inv.shape[0]),np.zeros(F_full.shape[0]-cov_inv.shape[0])])
                    D_vec[:trunc_size] = sing_val_sq
                #
 
            B = np.diag((1.0 + D_vec) ** (-1.0 / 2.0))
            #Computation of multi_dot([F_full, G.T,U,B.T,G_inv,F_full.T]) first by creating without F_full.T and multiply after by it.     
            AnoFt = np.linalg.multi_dot([np.multiply(F_full, np.diag(G_full)), np.multiply(np.multiply(U,np.diag(B)), np.diag(G_inv_full))])
            A = AnoFt.dot(F_full.T)
            # so overall: A = np.linalg.multi_dot([np.multiply(F_full, np.diag(G)), np.multiply(U,np.diag(B)), np.multiply(F_full,np.diag(G_inv)).T])
            Sigma_u = np.linalg.multi_dot([np.multiply(AnoFt,np.diag(Dp)),AnoFt.T])
            # so overall Sigma_u = np.linalg.multi_dot([A, Sigma, A.T])
            
        else:
            ### previous implementation
            start = time.perf_counter()
            Dp_vec = rtDp_vec**2 + self.joint_cov_noise 
            G = np.diag(np.sqrt(Dp_vec))
            G_inv = np.diag(1./np.sqrt(Dp_vec))
            U, D_vec, _ = la.svd(np.linalg.multi_dot([G.T, F.T, H.T, np.sqrt(cov_inv)]))
            D_vec = D_vec**2
            B = np.diag((1.0 + D_vec) ** (-1.0 / 2.0))
            A = np.linalg.multi_dot([F, 
                                    G.T, 
                                    U, 
                                    B.T, 
                                    G_inv, 
                                    F.T])
            
            Sigma_u = np.linalg.multi_dot([F, G.T, U, B.T, G_inv,
                                          np.diag(Dp_vec),
                                          G_inv, B, U.T, G, F.T])

            if zp.shape[0] < zp.shape[1]:
                F_u, Dp_u_vec , _ = randomized_svd(Sigma_u,
                                                        n_components=J-1,
                                                        n_iter=5,
                                                        random_state=None)

                F_u_full, _, _ = la.svd(np.multiply(F_u, Dp_u_vec))

                Dp_u_vec_full = np.ones(F_u_full.shape[0]) * Dp_u_vec[-1]
                Dp_u_vec_full[:J-1] = Dp_u_vec
                Dp_u = np.diag(Dp_u_vec_full)
                Sigma_u = np.linalg.multi_dot([F_u_full, Dp_u, F_u_full.T])
            end = time.perf_counter()
            print("Time for second SVD: ", end-start)
            
        # compute np.linalg.multi_dot([F_full, inv(Dp), F_full.T])
        Sigma_inv = np.linalg.multi_dot([np.multiply(F_full,1/np.diag(Dp)), F_full.T])
        
        zu_bar = np.dot(Sigma_u, \
                           (np.dot(Sigma_inv, zp_bar) + np.linalg.multi_dot([ np.multiply(H.T,np.diag(cov_inv)), x_t])))

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
