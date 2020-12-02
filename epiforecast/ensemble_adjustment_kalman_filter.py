import os
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import time
import warnings
 
class EnsembleAdjustmentKalmanFilter:

    def __init__(
            self,
            joint_cov_noise = 1e-2,
            obs_cov_noise = 1e-2,
            inflate_states = False,
            inflate_reg = 0.1,
            output_path=None):
        '''
        Instantiate an object that implements an Ensemble Adjustment Kalman Filter.

        Flags:
            * inflate_states: enable the inflation of states if True

        Key functions:
            * eakf.obs
            * eakf.update
            * eakf.compute_error
        Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        '''
        
        # Error
        self.error = np.empty(0)
        self.joint_cov_noise = joint_cov_noise
        self.obs_cov_noise = obs_cov_noise
        self.inflate_states = inflate_states
        self.inflate_reg = inflate_reg  # unit is in (%)
        self.output_path = output_path

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
            H_obs,
            scale=None,
            print_error=False,
            r=1.0,
            inflate_indices=None,
            save_matrices=False,
            save_matrices_name=None):

        '''
        - ensemble_state (np.array): J x M of update states for each of the J ensembles

        - clinical_statistics (np.array): transition rate model parameters for each of the J ensembles

        - transmission_rates (np.array): transmission rate of model parameters for each of the J ensembles

        - truth (np.array): M x 1 array of observed states.

        - cov (np.array): M x M array of covariances that represent observational uncertainty.
                          For example, an array of 0's represents perfect certainty.
                          Off-diagonal elements represent the fact that observations of state
                          i may not be independent from observations of state j. For example, this
                          can occur when a test applied to person ni alters the certainty of a subsequent
                          test to person nj.
                          We assume the covariance is diagonal in this code.

        #TODO: how to deal with no transition and/or transmission rates. i.e empty array input.
               (Could we just use an ensemble sized column of zeros? then output the empty array
                '''

        assert (truth.ndim == 1), 'EAKF init: truth must be 1d array'
        assert (cov.ndim == 2), 'EAKF init: covariance must be 2d array'
        assert (truth.size == cov.shape[0] and truth.size == cov.shape[1]),\
            'EAKF init: truth and cov are not the correct sizes'
        output_path = self.output_path

        # Observation data statistics at the observed nodes
        x_t = truth
        cov = r**2 * cov

        if scale is None: 
            cov = np.clip((1./np.maximum(x_t, 1e-12)/np.maximum(1-x_t, 1e-12)), -5, 5)**2 * cov
            x_t = np.log(np.maximum(x_t, 1e-12)/np.maximum(1.-x_t, 1e-12))

        #add reg to observations too
        cov = cov + np.diag(self.obs_cov_noise*np.ones(cov.shape[0]))

        try:
            # We assume independent variances (i.e diagonal covariance)
            cov_inv = np.diag(1/np.diag(cov))
        except np.linalg.linalg.LinAlgError:
            print('cov not invertible')
            cov_inv = np.ones(cov.shape)

        if save_matrices:
            save_cov_file =os.path.join(output_path, 'cov_matrix'+save_matrices_name+'.npy')
            np.save(save_cov_file, cov)
                
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
        xt = truth.size #observations over all times
        
        zp_bar = np.mean(zp, 0)
        
        if save_matrices:
            save_state_file =os.path.join(output_path, 'state_matrix'+save_matrices_name+'.npy')
            np.save(save_state_file, (zp-zp_bar).T)

               
        # Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        # Performing the first SVD of EAKF
        svd_failed = False
        num_svd_attempts = 0

        # Whittaker Hamill 01 Inflation prior to update
        zp_inflated = self.inflate_reg * (zp - zp_bar) + zp_bar
        zp[:,inflate_indices] = zp_inflated[:,inflate_indices]

        #observation operators:
        # full state to data:
        H = np.hstack([np.zeros((xt, pqs)), H_obs])
        Hpq = np.hstack([np.eye(pqs), np.zeros((pqs, xs))])
        Hs = np.hstack([np.zeros((xs, pqs)), np.eye(xs)])
        
        if save_matrices:
            save_H_file =os.path.join(output_path, 'H_matrix'+save_matrices_name+'.npy')
            np.save(save_H_file, H)
        

        # if ensemble_size < observations size, we pad the singular value matrix with added noise
        if zp.shape[0] < zp.shape[1]:    
            
            try:
                F_full, rtDp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
            except:
                print("First SVD not converge!", flush=True)
                np.save(os.path.join(output_path, 'svd_matrix_1.npy'),
                        (zp-zp_bar).T)
                svd_failed = True
            while svd_failed == True:
                num_svd_attempts = num_svd_attempts+1
                np.random.seed(num_svd_attempts*100)
                try:
                    svd_failed = False
                    F_full, rtDp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
                except:
                    svd_failed = True 
                    print("First SVD not converge!",flush=True)
            F = F_full[:,:J-1]
            rtDp_vec = rtDp_vec[:-1]
            rtDp_vec = 1./np.sqrt(J-1) * rtDp_vec
            rtDp_vec_full = np.zeros(zp.shape[1])
            rtDp_vec_full[:J-1] = rtDp_vec
            reg = 1
            sum_Dp_vec = sum(rtDp_vec**2)
            while (sum(rtDp_vec[:reg]**2) <= self.joint_cov_noise*sum_Dp_vec) and (reg < J-1):
                reg+=1
            
            
            if reg < J-1:
                print("reg value ",rtDp_vec[reg]**2," at index ",reg, " achieves threshold",sum(rtDp_vec[:reg]**2) / sum_Dp_vec)
                
                Dp_vec_full = rtDp_vec_full**2 + rtDp_vec_full[reg]**2*np.ones(rtDp_vec_full.shape)
            else:
                print("reg value lower than all evalues using 0.1*min(eval>0)")
                Dp_vec_full = rtDp_vec_full**2 
                Dp_vec_full[reg:] = rtDp_vec_full[reg:]**2 + 0.1*(rtDp_vec.min()**2)*np.ones(rtDp_vec_full[reg:].shape )  
            
            Dp = np.diag(Dp_vec_full)
        
        else:   
            try:
                F_full, rtDp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
            except:
                print("First SVD not converge!", flush=True)
                np.save(os.path.join(output_path, 'svd_matrix_1.npy'),
                        (zp-zp_bar).T)
                svd_failed = True
            while svd_failed == True:
                num_svd_attempts = num_svd_attempts+1
                np.random.seed(num_svd_attempts*100)
                try:
                    svd_failed = False
                    F_full, rtDp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
                except:
                    svd_failed = True 
                    print("First SVD not converge!")
            F = F_full
           
            rtDp_vec = 1./np.sqrt(J-1) * rtDp_vec
            reg = 1
            sum_Dp_vec = sum(rtDp_vec**2)
            while (sum(rtDp_vec[:reg]**2) <= self.joint_cov_noise*sum_Dp_vec) and (reg < J-1):
                reg+=1
            
            if reg < J-1:
                print("reg value ",rtDp_vec[reg]**2," at index ",reg, " achieves threshold",sum(rtDp_vec[:reg]**2) / sum_Dp_vec)
                Dp_vec_full = rtDp_vec**2 + rtDp_vec[reg]**2*np.ones(rtDp_vec.shape)
            
            else:
                print("reg value lower than all evalues")
                Dp_vec_full = rtDp_vec**2
                Dp_vec_full[-1] += rtDp_vec[-1]**2   
           
            Dp = np.diag(Dp_vec_full)

        # compute np.linalg.multi_dot([F_full, Dp, F_full.T])            
        #Sigma = np.linalg.multi_dot([np.multiply(F_full, np.diag(Dp)),F_full.T])
        
        G_full = np.diag(np.sqrt(Dp_vec_full))
        G_inv_full = np.diag(1./np.sqrt(Dp_vec_full))
            
        # Performing the second SVD of EAKF in the full space
        # computation of multidot([G_full.T, F_full.T, H.T, np.sqrt(cov_inv)])
        svd_failed = False
        num_svd_attempts = 0
        try:
            U, rtD_vec, _ = la.svd(np.linalg.multi_dot([np.multiply(F_full,np.diag(G_full)).T, np.multiply(H.T, np.sqrt(np.diag(cov_inv)))]), \
                                   full_matrices=True)
        except:
            print("Second SVD not converge!", flush=True)
            np.save(os.path.join(output_path, 'svd_matrix_2.npy'), \
                    np.linalg.multi_dot([np.multiply(F_full,np.diag(G_full)).T, \
                    np.multiply(H.T, np.sqrt(np.diag(cov_inv)))]))
            U, rtD_vec, _ = la.svd(np.linalg.multi_dot([np.multiply(F_full,np.diag(G_full)).T, np.multiply(H.T, np.sqrt(np.diag(cov_inv)))]), \
                               full_matrices=True)
            svd_failed = True
        while svd_failed == True:
            num_svd_attempts = num_svd_attempts+1
            print(num_svd_attempts, flush=True)
            np.random.seed(num_svd_attempts*100)
            try:
                svd_failed = False
                U, rtD_vec, _ = la.svd(np.linalg.multi_dot([np.multiply(F_full,np.diag(G_full)).T, np.multiply(H.T, np.sqrt(np.diag(cov_inv)))]), \
                                   full_matrices=True)
            except:
                svd_failed = True 
                print("Second SVD not converge!", flush=True)
        D_vec = np.zeros(F_full.shape[0])
        D_vec[:cov_inv.shape[0]] = rtD_vec**2

        B = np.diag((1.0 + D_vec) ** (-1.0 / 2.0))
        #Computation of multi_dot([F_full, G.T,U,B.T,G_inv,F_full.T]) first by creating without F_full.T and multiply after by it.     
        AnoFt = np.linalg.multi_dot([np.multiply(F_full, np.diag(G_full)), np.multiply(np.multiply(U,np.diag(B)), np.diag(G_inv_full))])
        A = AnoFt.dot(F_full.T)
        # so overall: A = np.linalg.multi_dot([np.multiply(F_full, np.diag(G)), np.multiply(U,np.diag(B)), np.multiply(F_full,np.diag(G_inv)).T])
        Sigma_u = np.linalg.multi_dot([np.multiply(AnoFt,np.diag(Dp)),AnoFt.T])
        # so overall Sigma_u = np.linalg.multi_dot([A, Sigma, A.T])

        # compute np.linalg.multi_dot([F_full, inv(Dp), F_full.T])           
        Sigma_inv = np.linalg.multi_dot([np.multiply(F_full,1/np.diag(Dp)), F_full.T])
        
        zu_bar = np.dot(Sigma_u, \
                           (np.dot(Sigma_inv, zp_bar) + np.linalg.multi_dot([ np.multiply(H.T,np.diag(cov_inv)), x_t])))

        # Update parameters and state in `zu`
        zu = np.dot(zp - zp_bar, A.T) + zu_bar

        # Store updated parameters and states
        x_logit = np.dot(zu, Hs.T)

        # Avoid overflow for exp
        x_logit = np.minimum(x_logit, 1e2)
        
        # replace unchanged states
        new_ensemble_state = np.exp(x_logit)/(np.exp(x_logit) + 1.0)
        
        if save_matrices:
            save_state_file =os.path.join(output_path, 'updated_state_matrix'+save_matrices_name+'.npy')
            np.save(save_state_file, (zu-zu_bar).T)

        pqout=np.dot(zu,Hpq.T)
        new_clinical_statistics = pqout[:, :clinical_statistics.shape[1]]
        new_transmission_rates  = pqout[:, clinical_statistics.shape[1]:]

        if (ensemble_state.ndim == 1):
            new_ensemble_state=new_ensemble_state.squeeze()

        # Compute error
        if print_error:
            self.compute_error(np.dot(x_logit, H_obs.T),x_t,cov)

        return new_ensemble_state, new_clinical_statistics, new_transmission_rates
