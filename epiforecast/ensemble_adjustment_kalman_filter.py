import numpy as np
import scipy.linalg as la
import sys

class EnsembleAdjustmentKalmanFilter:

    def __init__(self, full_svd = True, \
                 params_cov_noise = 1e-2, states_cov_noise = 1e-2, \
                 params_noise_active = True, states_noise_active = True):
        '''
        Instantiate an object that implements an Ensemble Adjustment Kalman Filter.

        Key functions:
            * eakf.obs
            * eakf.update
            * eakf.compute_error
        Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        '''

        if full_svd != True and full_svd != False:
            sys.exit("Incorrect flag detected for full_svd (needs to be True/False)!")

        # Error
        self.error = np.empty(0)
        self.full_svd = full_svd
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
    def update(self, ensemble_state, clinical_statistics, transmission_rates, truth, cov, r=1.0):

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

        # print("----------------------------------------------------")
        # print(x_t[:3])
        # print(" ")
        # print(np.diag(cov)[:3])
        # print("----------------------------------------------------")

        cov = (1./np.maximum(x_t, 1e-12)/np.maximum(1-x_t, 1e-12))**2 * cov
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
            params_noise_active=False

        # Ensemble size
        J = x.shape[0]

        # Sizes of q and x
        pqs = q[0].size +p[0].size
        xs = x[0].size

        zp_bar = np.mean(zp, 0)
        Sigma = np.cov(zp.T)

        #if only one state is given
        if Sigma.ndim < 2:
            Sigma=np.array([Sigma])
            Sigma=Sigma[np.newaxis]

        if self.full_svd == True:
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
            Sigma_inv = np.linalg.inv(Sigma)
            G = np.diag(np.sqrt(Dp_vec))
            G_inv = np.diag(1./np.sqrt(Dp_vec))
            U, D_vec, _ = la.svd(np.linalg.multi_dot([G.T, F.T, H.T, cov_inv, H, F, G]))
            B = np.diag((1.0 + D_vec) ** (-1.0 / 2.0))
            A = np.linalg.multi_dot([F, \
                                     G.T, \
                                     U, \
                                     B.T, \
                                     G_inv, \
                                     F.T])
            Sigma_u = np.linalg.multi_dot([A, Sigma, A.T])

        else:

            from sklearn.decomposition import TruncatedSVD

            # Follow Anderson 2001 Month. Weath. Rev. Appendix A.
            # Preparing matrices for EAKF
            H = np.hstack([np.zeros((xs, pqs)), np.eye(xs)])
            Hpq = np.hstack([np.eye(pqs), np.zeros((pqs, xs))])
            svd1 = TruncatedSVD(n_components=J-1, random_state=42)
            svd1.fit(Sigma)
            F = svd1.components_.T
            Dp_vec = svd1.singular_values_
            Dp = np.diag(Dp_vec)
            Sigma_inv = np.linalg.multi_dot([F, np.linalg.inv(Dp), F.T])
            G = np.diag(np.sqrt(Dp_vec))
            G_inv = np.diag(1./np.sqrt(Dp_vec))
            U, D_vec, _ = la.svd(np.linalg.multi_dot([G.T, F.T, H.T, cov_inv, H, F, G]))
            B = np.diag((1.0 + D_vec) ** (-1.0 / 2.0))
            A = np.linalg.multi_dot([F, \
                                     G.T, \
                                     U, \
                                     B.T, \
                                     G_inv, \
                                     F.T])
            Sigma_u = np.linalg.multi_dot([A, Sigma, A.T])

            # Adding noises (model uncertainties) to the truncated dimensions
            # Need to further think about how to reduce the cost of full SVD here
            #F_u, Dp_u_vec, _ = la.svd(Sigma_u)
            #Dp_u_vec[J-1:] = np.min(Dp_u_vec[:J-1])
            #Sigma_u = np.linalg.multi_dot([F_u, np.diag(Dp_u_vec), F_u.T])

            # Adding noises approximately to the truncated dimensions (Option #1)
            svd1.fit(Sigma_u)
            noises = np.identity(Sigma_u.shape[0]) * svd1.singular_values_[-1]
            Sigma_u = Sigma_u + noises

            # Adding noises approximately to the truncated dimensions (Option #2)
            #svd1.fit(Sigma_u)
            #vec = np.diag(Sigma_u)
            #vec = np.maximum(vec, svd1.singular_values_[-1])
            ##vec = np.maximum(vec, np.sort(vec)[-J])
            #np.fill_diagonal(Sigma_u, vec)

        ## Adding noises into data for each ensemble member (Currently deactivated)
        # noise = np.array([np.random.multivariate_normal(np.zeros(xs), cov) for _ in range(J)])
        # x_t = x_t + noise
        zu_bar = np.dot(Sigma_u, \
                           (np.dot(Sigma_inv, zp_bar) + np.dot(np.dot(H.T, cov_inv), x_t)))

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
        self.compute_error(x_logit,x_t,cov)

        #print("new_clinical_statistics", new_clinical_statistics)
        #print("new_transmission_rates", new_transmission_rates)
        return new_ensemble_state, new_clinical_statistics, new_transmission_rates
