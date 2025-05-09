from typing import Optional
import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.sparse as sps

from . import kernel
from .utils import Diag


def Kernel_SE_2dim(x1,x2,y1,y2,l1,l2):
    X1,X2 = np.meshgrid(x1,x2,indexing='ij')
    Y1,Y2 = np.meshgrid(y1,y2,indexing='ij')

    return np.exp(-0.5*((X1-X2)**2/l1**2 + (Y1-Y2)**2/l2**2))


def log_det(A,scale=1):
    try:
        A = scipy.linalg.cholesky(A)
        return np.sum(np.log(np.diag(A)))*2 + A.shape[0]*np.log(scale)
    except:
        lam,_ = np.linalg.eigh(A)

        index = lam>0
        return np.sum(np.log(lam[index])) + np.sum(index)*np.log(scale)

class GPT_log_general:
    def __init__(self,
        H:np.ndarray,
        r_idc:np.ndarray,
        z_idc:np.ndarray,
        length_scale,
        eps:float = 1e-6,
        ) -> None:
        """
        GPT_log_general class for log-Gaussian Process Tomography.      

        Parameterse
        ----------
        H : np.ndarray
            geometry matrix of shape (nI,ng).
        r_idc : np.ndarray
            radial values of the inducing points with shape (nI,).
        z_idc : np.ndarray
            vertical values of the inducing points with shape (nI,).
        length_scale_idc : float
            length scale of the kernel.
        eps : float, optional
            regularization parameter for the kernel invrsion. The default is 1e-6.
        """
        self.H = H
        self.rI = r_idc
        self.zI = z_idc

        self.nI = z_idc.size
        self.ng = H.shape[0]
        self.length = length_scale
        self.regularization = eps

        self.KII = Kernel_SE_2dim(r_idc,r_idc,z_idc,z_idc,length_scale,length_scale)
        self.normalized = False

    def set_kernel_and_boudary(self,
        rb:np.ndarray,
        zb:np.ndarray,
        bound_sig = 0,
        bound_value = -5,
        ):
        """"
        Set the kernel and boundary conditions for the Gaussian process.

        Parameters
        ----------
        rb : np.ndarray
            radial values of the boundary points with shape (nb,).
        zb : np.ndarray
            vertical values of the boundary points with shape (nb,).
        bound_sig : float, optional
            variance on the boundary. The default is 0.
        bound_value : float, optional
            value on the boundary. The default is -5.
        """
        length = self.length 
        self.rb = rb
        self.zb = zb
        self.nb = zb.size
        Kernel_IB = Kernel_SE_2dim(self.rI,rb,self.zI,zb,length,length)
        Kernel_bb = Kernel_SE_2dim(rb,rb,zb,zb,length,length)

        if (bound_sig < 0) | (bound_sig >= 1):
            raise ValueError('bound_sig must be non-negative')
        
        factor = 1/(1-bound_sig**2)    
        Kernel_bb = factor*Kernel_bb
        Kernel_bb_inv = np.linalg.inv(Kernel_bb + 1e-6*np.eye(Kernel_bb.shape[0]))
        
        f_b = Kernel_IB @ (Kernel_bb_inv @ (bound_value*np.ones(self.nb)) )

        K_pri = self.KII - Kernel_IB @ (Kernel_bb_inv @ Kernel_IB.T)

        self.K_pri = K_pri
        self.f_pri = f_b

        self.K_inv = np.linalg.inv(K_pri+self.regularization*np.eye(self.nI))
        
        self.log_det_K = log_det(self.K_pri+self.regularization*np.eye(self.nI))

    def set_obs(self,
        sig_ratio  :np.ndarray,
        g_obs   :np.ndarray,
        normalize :bool =False,
        ):
        """
        Set the signal and observation data for the Gaussian process.

        Parameters
        ----------
        sig_ratio : np.ndarray
            sigma ratio of the observations with shape (ng,).
        g_obs : np.ndarray
            observation data with shape (ng,).
        normalize : bool, optional
            whether to normalize the data. The default is False.
        """

        self.g_obs = g_obs
        self.normalize(normalize)  
        
            
        self.sig_inv = 1/sig_ratio  

        Hn = self.H /self.H_scale

        self.Sigi_obs = self.sig_inv* ( self.g_obs_n )
        self.sigiH = sps.diags(self.sig_inv) @ Hn 

        #self.Hsig2iH  = (self.sigiH.T @ self.sigiH).toarray() 
        # self.Hsig2iH = sparse_dot_mkl.gram_matrix_mkl(sigiH_t,transpose=True,dense=True)　#2時間溶かした戦犯
        self.Hsig2iH = self.sigiH.T @self.sigiH
        self.log_det_Sig = 2*np.sum(np.log(sig_ratio))

    def normalize(self,
        is_normalize:bool = False
        ):
        if is_normalize and not self.normalized:
            self.H_scale = np.mean(self.H@np.ones(self.nI))
            self.g_scale = self.g_obs.mean()
            self.g_obs_n = 1/self.g_scale * self.g_obs

            self.f_scale = self.g_scale/self.H_scale
            self.normalized = True

        else:
            self.H_scale = 1
            self.g_scale = 1
            self.f_scale = 1
            self.normalized = False
            self.g_obs_n = self.g_obs
            pass

    
    def update(self,
        f:np.ndarray,
        sig_scale  = None,
        ) -> tuple[np.ndarray,float]:
        """"
        Update for Laplace approximation.

        Parameters
        ----------
        f : np.ndarray
            latent function values with shape (nI,).
        sig_scale : float, optional
            scale of the sigma. The default is 0.1.

        Returns
        -------
        delta_f : np.ndarray
            update direction for the latent function values with shape (nI,).
        loss : float
            loss value for the intermediate step.
        """

        if sig_scale is not None:
            self.sig_scale = sig_scale
        
        r_f = f - self.f_pri
        exp_f = np.exp(f)
        fxf = np.einsum('i,j->ij',exp_f,exp_f)
        
        SiR = self.sigiH @ exp_f - self.Sigi_obs
        c1 = 1/self.sig_scale**2 *(self.sigiH.T @ SiR) * exp_f #self.sigiH.T @ SiR =  self.Hsig2iH  @ exp_f -  self.sigiH.T@ self.Sigi_obs
        C1 = 1/self.sig_scale**2 *self.Hsig2iH * fxf 

        self.SiR = SiR
        self.r_f = r_f

        Psi_df   = -c1 - self.K_inv @ r_f 

        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv

        LPsi = Psi_dfdf
        DPsi = Psi_df

        loss = abs(DPsi).mean()

        delta_f = - np.linalg.solve(LPsi,DPsi)

        delta_f[delta_f<-3] = -3
        delta_f[delta_f>+3] = +3

        return delta_f,loss
    
    def postprocess(self,
        f:np.ndarray,
        ):
        """
        Postprocess for Laplace approximation.

        Parameters
        ----------
        f : np.ndarray
            latent function values with shape (nI,).
        """

        
        exp_f = np.exp(f)
        fxf = np.einsum('i,j->ij',exp_f,exp_f)
        
        
        SiR = self.sigiH @ exp_f - self.Sigi_obs
        c1 = 1/self.sig_scale**2 *(self.sigiH.T @ SiR) * exp_f #self.sigiH.T @ SiR =  self.Hsig2iH  @ exp_f -  self.sigiH.T@ self.Sigi_obs
        C1 = 1/self.sig_scale**2 *self.Hsig2iH * fxf 


        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv

        DPsi = Psi_dfdf
        self.Kf_pos_inv = -DPsi
        self.Kf_pos     = np.linalg.inv(self.Kf_pos_inv)
        self.sigf_pos = np.sqrt(np.diag(self.Kf_pos))
        self.f_mean =f
        self.f_mean = self.f_mean + np.log(self.f_scale)


        self.log_det_Kpos = log_det(self.Kf_pos)
        self.loss_g = 1/self.sig_scale**2 *np.dot(self.SiR, self.SiR)
        self.loss_f =  self.r_f@ (self.K_inv @ self.r_f) 

        self.log_det_Sig2 = self.log_det_Sig + 2*np.log(self.sig_scale)*self.ng


        self.mll = (-self.loss_g -self.log_det_Sig2) -self.loss_f  - self.log_det_K + self.log_det_Kpos
        self.mll = 0.5*self.mll- 0.5*self.ng*np.log(2*np.pi)

        pass



    def expf_mean(self,):
        return np.exp(self.f_mean+0.5*self.sigf_pos**2)
    
    
    def expf_median(self,):
        return np.exp(self.f_mean)
    
    def expf_std(self):
        return np.sqrt(np.exp(2*self.f_mean+ self.sigf_pos**2)*(np.exp(self.sigf_pos**2)-1))
 
class GPT_log_general:
    def __init__(self,
        H:np.ndarray,
        Kf_pri:np.ndarray,
        muf_pri:np.ndarray,
        eps:float = 1e-6,
        ) -> None:
        """
        GPT_log_general class for log-Gaussian Process Tomography.      

        Parameterse
        ----------
        H : np.ndarray
            geometry matrix of shape (nI,ng).
        Kf_pri : np.ndarray
            prior covariance matrix of shape (nI,nI).
        muf_pri : np.ndarray
            prior mean vector of shape (nI,).
        eps : float, optional
        """
        self.H = H

        self.nI = H.shape[1]
        self.ng = H.shape[0]
        self.regularization = eps
        self.Kf_pri = 0.5*(Kf_pri+Kf_pri.T)
        self.muf_pri = muf_pri
        self.normalized = False

        
        self.K_inv = np.linalg.inv(self.Kf_pri+self.regularization*np.eye(self.nI))
        self.log_det_K = log_det(self.Kf_pri+self.regularization*np.eye(self.nI))


    def set_obs(self,
        g_obs   :np.ndarray,
        sig_ratio  :np.ndarray=None,
        normalize :bool =False,
        sig_scale:float = 1,
        ):
        """
        Set the signal and observation data for the Gaussian process.
        we models the likelihood for residuals as:
        $g = Hf + \epsilon$ where $\epsilon ~ N (0, \Sigma_g)$
        $\Sigma_g = (sig_scale *Diag(sig_ratio))**2$

        Parameters
        ----------
        g_obs : np.ndarray
            observation data with shape (ng,).
        sig_ratio : np.ndarray
            sigma ratio of the observations with shape (ng,).
        sig_scale : float, optional
            scale of the sigma. The default is 1.
        normalize : bool, optional
            whether to normalize the data. The default is False.
        """

        self.g_obs = g_obs
        self.normalized = not normalize
        self.normalize(normalize)  

        self.sig_scale = sig_scale
        
        if sig_ratio is None:
            sig_ratio = np.ones(self.ng)
        self.sig_inv = 1/sig_ratio  

        Hn = self.H /self.H_scale

        self.Sigi_obs = self.sig_inv* ( self.g_obs_n )
        self.sigiH = sps.diags(self.sig_inv) @ Hn 

        #self.Hsig2iH  = (self.sigiH.T @ self.sigiH).toarray() 
        # self.Hsig2iH = sparse_dot_mkl.gram_matrix_mkl(sigiH_t,transpose=True,dense=True)　#2時間溶かした戦犯
        self.Hsig2iH = self.sigiH.T @self.sigiH
        self.log_det_Sig = 2*np.sum(np.log(sig_ratio))

    def normalize(self,
        is_normalize:bool = False
        ):
        if is_normalize and not self.normalized:
            self.H_scale = np.mean(self.H@np.ones(self.nI))
            self.g_scale = self.g_obs.mean()
            self.g_obs_n = 1/self.g_scale * self.g_obs

            self.f_scale = self.g_scale/self.H_scale
            self.normalized = True

        else:
            self.H_scale = 1
            self.g_scale = 1
            self.f_scale = 1
            self.normalized = False
            self.g_obs_n = self.g_obs
            pass

    
    def update(self,
        f:np.ndarray,
        sig_scale  = None,
        ) -> tuple[np.ndarray,float]:
        """"
        Update for Laplace approximation.

        Parameters
        ----------
        f : np.ndarray
            latent function values with shape (nI,).
        sig_scale : float, optional
            scale of the sigma. The default is 0.1.

        Returns
        -------
        delta_f : np.ndarray
            update direction for the latent function values with shape (nI,).
        loss : float
            loss value for the intermediate step.
        """

        if sig_scale is not None:
            self.sig_scale = sig_scale
        
        r_f = f - self.muf_pri
        exp_f = np.exp(f)
        fxf = np.einsum('i,j->ij',exp_f,exp_f)
        
        SiR = self.sigiH @ exp_f - self.Sigi_obs
        c1 = 1/self.sig_scale**2 *(self.sigiH.T @ SiR) * exp_f #self.sigiH.T @ SiR =  self.Hsig2iH  @ exp_f -  self.sigiH.T@ self.Sigi_obs
        C1 = 1/self.sig_scale**2 *self.Hsig2iH * fxf 

        self.SiR = SiR
        self.r_f = r_f

        Psi_df   = -c1 - self.K_inv @ r_f 

        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv

        LPsi = Psi_dfdf
        DPsi = Psi_df

        loss = abs(DPsi).mean()

        delta_f = - np.linalg.solve(LPsi,DPsi)

        delta_f[delta_f<-3] = -3
        delta_f[delta_f>+3] = +3

        return delta_f,loss
    
    def postprocess(self,
        f:np.ndarray,
        ):
        """
        Postprocess for Laplace approximation.

        Parameters
        ----------
        f : np.ndarray
            latent function values with shape (nI,).
        """

        
        exp_f = np.exp(f)
        fxf = np.einsum('i,j->ij',exp_f,exp_f)
        
        
        SiR = self.sigiH @ exp_f - self.Sigi_obs
        c1 = 1/self.sig_scale**2 *(self.sigiH.T @ SiR) * exp_f #self.sigiH.T @ SiR =  self.Hsig2iH  @ exp_f -  self.sigiH.T@ self.Sigi_obs
        C1 = 1/self.sig_scale**2 *self.Hsig2iH * fxf 


        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv

        DPsi = Psi_dfdf
        self.Kf_pos_inv = -DPsi
        self.Kf_pos     = np.linalg.inv(self.Kf_pos_inv)
        self.sigf_pos = np.sqrt(np.diag(self.Kf_pos))
        self.f_mean =f
        self.f_mean = self.f_mean + np.log(self.f_scale)


        self.log_det_Kpos = log_det(self.Kf_pos)
        self.loss_g = 1/self.sig_scale**2 *np.dot(self.SiR, self.SiR)
        self.loss_f =  self.r_f@ (self.K_inv @ self.r_f) 

        self.log_det_Sig2 = self.log_det_Sig + 2*np.log(self.sig_scale)*self.ng


        self.mll = (-self.loss_g -self.log_det_Sig2) -self.loss_f  - self.log_det_K + self.log_det_Kpos
        self.mll = 0.5*self.mll- 0.5*self.ng*np.log(2*np.pi)

        pass


    @property
    def expf_mean(self,):
        return np.exp(self.f_mean+0.5*self.sigf_pos**2)
    
    @property
    def expf_median(self,):
        return np.exp(self.f_mean)
    @property
    def expf_std(self):
        return np.sqrt(np.exp(2*self.f_mean+ self.sigf_pos**2)*(np.exp(self.sigf_pos**2)-1))   

class GPT_log:
    def __init__(self,
        H:np.ndarray,
        eps:float = 1e-6,
        ) -> None:
        """
        GPT_log class for log-Gaussian Process Tomography.

        Parameterse
        ----------
        H : np.ndarray
            geometry matrix of shape (nI,ng).
        r_idc : np.ndarray
            radial values of the inducing points with shape (nI,).
        z_idc : np.ndarray
            vertical values of the inducing points with shape (nI,).
        length_scale_idc : float
            length scale of the kernel.
        eps : float, optional
            regularization parameter for the kernel invrsion. The default is 1e-6.
        """
        
        self.H = H
        self.ng = H.shape[0]
        self.regularization = eps

    def set_prior(self,
        Kf_pri:np.ndarray,
        muf_pri:np.ndarray,
        ):
        """
        Set the prior for the Gaussian process. 

        Parameters
        ----------
        Kf_pri : np.ndarray
            prior covariance matrix of shape (nI,nI).
        muf_pri : np.ndarray
            prior mean vector of shape (nI,).
        """

        self.Kf_pri = 0.5*(Kf_pri+Kf_pri.T)
        self.muf_pri = muf_pri

        self.Kf_pri_inv = np.linalg.inv(self.Kf_pri+self.regularization*np.eye(self.Kf_pri.shape[0]))
        self.log_det_Kf_pri = log_det(self.Kf_pri+self.regularization*np.eye(self.Kf_pri.shape[0]))

    def set_obs(self,
        sig_ratio  :np.ndarray,
        g_obs   :np.ndarray,
        normalize :bool =False,
        sig_scale:float = None,
        ):
        """
        Set the signal and observation data for the Gaussian process.

        Parameters
        ----------
        sig_ratio : np.ndarray
            sigma ratio of the observations with shape (ng,).
            $\epsilon ~ N (0, \Sigma_g)$
            $\Sigma_g = sigma_scale * Diag(sig_ratio) $
        g_obs : np.ndarray
            observation data with shape (ng,).
        normalize : bool, optional
            whether to normalize the data. The default is False.
        """

        self.g_obs = g_obs
        self.normalize(normalize)  

        if sig_scale is None:
            sig_scale = 1
        else:
            self.sig_scale = sig_scale
        
            
        self.Sig_inv = Diag(sig_ratio).inv()

        Hn = self.H /self.H_scale

        self.Sigi_obs = self.Sig_inv@ self.g_obs_n 
        self.sigiH = self.Sig_inv @ Hn 

        #self.Hsig2iH  = (self.sigiH.T @ self.sigiH).toarray() 
        # self.Hsig2iH = sparse_dot_mkl.gram_matrix_mkl(sigiH_t,transpose=True,dense=True)　#2時間溶かした戦犯
        self.Hsig2iH = self.sigiH.T @self.sigiH

    
    def normalize(self,
        is_normalize:bool = False
        ):
        if is_normalize and not self.normalized:
            self.H_scale = np.mean(self.H@np.ones(self.nI))
            self.g_scale = self.g_obs.mean()
            self.g_obs_n = 1/self.g_scale * self.g_obs

            self.f_scale = self.g_scale/self.H_scale
            self.normalized = True

        else:
            self.H_scale = 1
            self.g_scale = 1
            self.f_scale = 1
            self.normalized = False
            self.g_obs_n = self.g_obs
            pass
