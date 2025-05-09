import os
import sys

import json
import numpy as np
from typing import Callable, Optional, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import mpl_toolkits.axes_grid1
import numpy.typing as npt

from .. import plot_utils
import zray
from gpoloidal import kernel 
from . import mag

FILE_DIR = os.path.dirname(__file__)


class Kernel2D_scatter_rt1(kernel.Kernel2D_scatter):
    """
    RT-1のコイルの磁場を計算するクラス
    """

    def __init__(self):
        """
        """
        self.dict_path= os.path.join(FILE_DIR, 'rt1_simple_frame.json')
        _dict = json.load(open(self.dict_path, 'r'))
        vessel = zray.AxisymmetricVessel.from_dict(_dict)
        print(f"load vessel from {self.dict_path}")


        super().__init__(vessel=vessel)

    
    def set_flux_kernel(self,
                        
        psi_scale        : float = 0.3,
        B_scale          : float = 0.3,
        is_bound         : bool  = True ,
        bound_value      : float = 0,
        bound_sig        : float = 0.1,
        mean             : float = 0,
        separatrix       : bool = False,
        out_scale_of_kernel : float = 1,
        zero_value_index : npt.NDArray[np.bool_] |None = None, # requres b
        is_static_kernel : bool = False,
        babs_symmetry    : bool = True,
        eps : float = 1e-6  

        )->Tuple[np.ndarray,np.ndarray]:

        rI,zI = self.r_idc,self.z_idc
        psi_i = mag.psi(rI,zI,separatrix=separatrix)
        br_i,bz_i = mag.bvec(rI,zI,separatrix=separatrix)
        babs_i = np.sqrt(br_i**2+bz_i**2)
        
                
        Psi_i = np.meshgrid(psi_i,psi_i,indexing='ij')

        if babs_symmetry:
            logb_i = np.log(babs_i)
        else:
            #このpythonが存在するディレクトリのパスを取得
            file_name = os.path.join(os.path.dirname(__file__), 'Bmin_spline.pkl')
            import pickle
            with open(file_name, 'rb') as f:
                Bmin_spline = pickle.load(f)
            log_Bsq_min_i = Bmin_spline(psi_i)

            mirror = np.log(babs_i) - 0.5*log_Bsq_min_i

            eps2 = 1e-6

            br_p,bz_p = mag.bvec(rI+eps2*br_i/babs_i,zI+eps2*bz_i/babs_i)
            br_m,bz_m = mag.bvec(rI-eps2*br_i/babs_i,zI-eps2*bz_i/babs_i)
            bsq_p = br_p**2 + bz_p**2   
            bsq_m = br_m**2 + bz_m**2
            bsq_diff = (bsq_p - bsq_m)/(2*eps2)

            logb_i = np.where(bsq_diff > 0, mirror+ 1e-6, -mirror- 1e-6)

        logBabs_i = np.meshgrid(logb_i,logb_i,indexing='ij')
                
        psi_len  =psi_i.std()*psi_scale
        psi_psi = (Psi_i[0]-Psi_i[1])**2/ psi_len**2

        b_len = logb_i.std()*B_scale
        b_b   = (logBabs_i[0]-logBabs_i[1])**2/b_len**2
                
        Kflux_ii = np.exp(-0.5*(psi_psi+b_b))

        
        if not is_bound:
            mu_f_pri = mean*np.ones_like(self.r_idc)
            Kflux_pri = Kflux_ii
        
        else:
            if zero_value_index is None:
                index = np.zeros(self.nI,dtype=bool)
            else:
                index = zero_value_index
            factor = 1/ (1-bound_sig**2)
                    
            rb,zb = self.r_bd,self.z_bd
            zo,ro = np.concatenate([zI[index],zb]), np.concatenate([rI[index],rb])
                    
            psi_o = mag.psi(ro,zo,separatrix=separatrix)
            br_o,bz_o = mag.bvec(ro,zo,separatrix=separatrix)
            babs_o = np.sqrt(br_o**2+bz_o**2)

            if babs_symmetry:
                logb_o = np.log(babs_o)
            else:
                log_Bsq_min_o = Bmin_spline(psi_o)
                mirror = np.log(babs_o) - 0.5*log_Bsq_min_o

                eps2 = 1e-6

                br_p,bz_p = mag.bvec(ro+eps2*br_o/babs_o,zo+eps2*bz_o/babs_o)
                br_m,bz_m = mag.bvec(ro-eps2*br_o/babs_o,zo-eps2*bz_o/babs_o)
                bsq_p = br_p**2 + bz_p**2   
                bsq_m = br_m**2 + bz_m**2
                bsq_diff = (bsq_p - bsq_m)/(2*eps2)

                logb_o = np.where(bsq_diff > 0, mirror+ 1e-6, -mirror- 1e-6)
                    

            Psi_o = np.meshgrid(psi_o,psi_o,indexing='ij') / psi_len
            lnBabs_o = np.meshgrid(logb_o,logb_o,indexing='ij') / b_len
            Psi_io = np.meshgrid(psi_i,psi_o,indexing='ij') / psi_len
            lnBabs_io = np.meshgrid(logb_i,logb_o,indexing='ij') / b_len

            
            Kb_oo =  np.exp(-0.5*((Psi_o[0]-Psi_o[1])**2  +(lnBabs_o[0]-lnBabs_o[1])**2))
            Kb_io =  np.exp(-0.5*((Psi_io[0]-Psi_io[1])**2+(lnBabs_io[0]-lnBabs_io[1])**2))

            Kb_oo = factor*Kb_oo
            Kb_oo_inv = np.linalg.inv(Kb_oo+eps*np.eye(ro.size))            
            Kflux_pri = Kflux_ii - Kb_io @ Kb_oo_inv @ Kb_io.T 
            Kflux_pri = out_scale_of_kernel**2*Kflux_pri
            mu_f_pri  = mean+Kb_io @  (Kb_oo_inv @ (bound_value*np.ones(ro.size)-mean)  ) 

        
        if separatrix:
            #babs_min = babs_i.min()
            #around_spx = babs_i < 100*babs_min
            ## 行と列がround_spxのとき0を代入
            #Kflux_pri[around_spx,:] = 0
            #Kflux_pri[:,around_spx] = 0
            ## substitute 1 to diagonal elements of around_spx
            #Kflux_pri[around_spx,around_spx] = out_scale_of_kernel**2*bound_sig**2 + eps*0
            #mu_f_pri[around_spx] = bound_value
            pass

        if is_static_kernel:
            self.Kf_pri = Kflux_pri 
            self.muf_pri = mu_f_pri 
            self.kernel_type = 'flux kernel'
                
            self.Kf_pri_property = {
                #'kernel_type': self.kernel_type,
                'psi_scale'  : psi_scale,
                'B_scale'    : B_scale,
                'is_bound'   : is_bound ,
                'bound_value': bound_value,
                'bound_sig'  : bound_sig }
            

        return Kflux_pri,mu_f_pri
    

    def  plt_rt1_flux(self,
        ax:Optional[plt.Axes] = None,      
        separatrix:bool =True,
        is_inner:bool =False,
        append_frame :bool =True,
        **kwargs_contour,
        )->None:
        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal')
        R,Z = np.meshgrid(self.r_plot,self.z_plot,indexing='xy')
        Psi = mag.psi(R,Z,separatrix=separatrix)
        extent = self.im_kwargs['extent']
        origin = self.im_kwargs['origin']
        kwargs = {'levels':20,'colors':'black','alpha':0.3}
        kwargs.update(kwargs_contour)
        if is_inner:
            Psi = Psi*self.mask
        else:
            mpsi_max = -mag.psi(0.2,0.,separatrix=separatrix)
            mpsi_min = -mag.psi(self.r_plot.min(),self.z_plot.max(),separatrix=separatrix)
            kwargs['levels'] = np.linspace(mpsi_min,mpsi_max,kwargs['levels'],endpoint=False)
        
        ax.contour(-Psi,extent=extent,origin=origin,**kwargs)
        if append_frame:
            self.vessel.plot()

if __name__ == "__main__":

    ## Convert DXF to JSON
    dxf_path = os.path.join(FILE_DIR, 'rt1_simple_frame.dxf')
    vessel = zray.AxisymmetricVessel.load_from_dxf(dxf_path)
    vessel_dict = vessel.to_dict()

    filename = os.path.join(FILE_DIR, 'rt1_simple_frame.json')
    with open(filename, 'w') as f:
        json.dump(vessel_dict, f, indent=4)
    print(f"convert rt1 vessel to {filename} from {dxf_path}")