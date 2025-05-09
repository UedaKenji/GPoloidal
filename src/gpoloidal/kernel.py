import numpy as np
from typing import Callable, Optional, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import mpl_toolkits.axes_grid1
import numpy.typing as npt
import os,sys


import zray
from . import plot_utils

from . import geometry_matrix


class Kernel2D_scatter():
    def __init__(self,
        vessel: zray.vessel.AxisymmetricVessel,
        ) -> None:
        
        """
        import dxf file

        Parameters
        ----------
        dxf_file : str
            Path of the desired file.
        show_print : bool=True,
            print property of frames
        Note
        ----
        dxf_file is required to have units of (mm).
        """
        self.vessel = vessel    
        self.V = None
        #self.im_shape: Union[Tuple[int,int],None] = None

        print('you have to "create_induced_point()" or "load_point()" in advance!')

    def create_inducing_point(self,
        z_grid: np.ndarray,
        r_grid: np.ndarray,
        length_sq_fuction: Callable[[np.ndarray,np.ndarray],np.ndarray],
        factor: float = 1.0,
        ) -> Tuple[np.ndarray,np.ndarray] | None:     
        """
        create inducing point based on length scale function

        Parameters
        ----------
        z_grid: np.ndarray,
        r_grid: np.ndarray,
        length_sq_fuction: Callable[[float,float],None],

        Reuturns
        ----------
        zI: np.ndarray,
        rI: np.ndarray,  
        """
        
        if not 'r_bd'  in dir(self):
            print('set_bound() is to be done in advance!')
            return
        
        rr,zz = np.meshgrid(r_grid,z_grid)

        self.register_ls_sq_function(length_sq_fuction,factor)

        length_sq = self.length_scale_sq_func(rr,zz)
        mask, _ = self.vessel.detect_grid(r_grid=r_grid, z_grid=z_grid,static=False)


        mask = (np.nan_to_num(mask) == 1)

        rI, zI = np.zeros(1),np.zeros(1)
        rI[0], zI[0] = r_grid[0],z_grid[0]
        is_short = True
        for i, zi in enumerate(tqdm(z_grid)):
            for j, ri in enumerate(r_grid):
                if mask[i,j]:
                    db_min = d2min(ri,zi,self.r_bd, self.z_bd)

                    if rI.size < 500:
                        d2_min = d2min(ri,zi,rI,zI)
                    else:
                        d2_min = d2min(ri,zi,rI[-500:],zI[-500:])

                    if length_sq[i,j] > min(db_min,d2_min):
                        is_short = True
                    elif is_short:
                        is_short = False
                        rI = np.append(rI,ri)
                        zI = np.append(zI,zi)                    

        rI,zI = rI[1:], zI[1:]

        self.__z_idc, self.__r_idc = zI, rI
        self.__ls_idc = self.length_scale_func(self.r_idc,self.z_idc)
        self.__ls_sq_idc = self.length_scale_sq_func(self.r_idc,self.z_idc)
        self.__nI = rI.size

        print('num of induced point is ',self.nI)
    
    def register_ls_sq_function(self,
        length_scale_sq: Callable[[np.ndarray,np.ndarray],np.ndarray],
        factor = 1.0
        ) -> None:
        """

        register length scale function
        Parameters
        ----------

        length_scale_sq: Callable[[np.ndarray,np.ndarray],np.ndarray],

        """        
        self._length_scale_sq_func: Callable[[np.ndarray,np.ndarray],np.ndarray]=  length_scale_sq
        self.__factor = factor
        # self.__z_idcが定義されているかどうか？
        if hasattr(self, 'z_idc') and hasattr(self, 'r_idc'):
            self.__ls_idc = self.length_scale_func(self.r_idc,self.z_idc)
            self.__ls_sq_idc = self.length_scale_sq_func(self.r_idc,self.z_idc)


    @property
    def factor(self) -> float: return self.__factor
    @property
    def z_idc(self) -> np.ndarray: return self.__z_idc
    @property
    def r_idc(self) -> np.ndarray: return self.__r_idc
    @property
    def ls_idc(self) -> np.ndarray: return self.__ls_idc
    @property
    def ls_sq_idc(self) -> np.ndarray: return self.__ls_sq_idc
    @property
    def nI(self) -> int: return self.__nI

    def length_scale_sq_func(self,r:np.ndarray,z:np.ndarray) -> np.ndarray:  
        """
        length scale function

        Parameters
        ----------
        r: np.ndarray,
        z: np.ndarray,

        Reuturns
        ----------
        length scale function value at (r,z)
        """
        
        return self._length_scale_sq_func(r,z)*self.factor**2
        
    def length_scale_func(self,r:np.ndarray,z:np.ndarray) -> np.ndarray:
        return np.sqrt(self.length_scale_sq_func(r,z))



        
    def set_bound_arange(self,
        delta_l = 1e-2,
        ) -> tuple[np.ndarray,np.ndarray] :
        """
        create induced point with equal space 

        Parameters
        ----------
        delta_l: space length [m] 

        Reuturns
        ----------
        """

        z_all, r_all = np.zeros(0),np.zeros(0)
        for entity in self.vessel.Lines:
            #r0,r1 = entity.start[0]/1000, entity.end[0]/1000 
            #z0,z1 = entity.start[1]/1000, entity.end[1]/1000
            r0,z0 = entity.p0
            r1,z1 = entity.p1
            l = np.sqrt((z0-z1)**2 + (r0-r1)**2)
            n = int(l/delta_l) + 1 
            z = np.linspace(z0,z1,n)
            r = np.linspace(r0,r1,n)
            z_all = np.append(z_all,z)
            r_all = np.append(r_all,r)  

        for entity in self.vessel.Arcs:
            #angle = entity.end_angle- entity.start_angle

            angle = entity.theta_end - entity.theta_start
            angle = 360*( angle < 0 ) + angle 
            radius = entity.radius 
            n = int(radius*angle/180*np.pi/delta_l) + 1
            #print(n,angle)
            theta = np.linspace(entity.theta_start,entity.theta_start+angle,n) / 180*np.pi
            r = entity.center[0] + radius*np.cos(theta)
            z = entity.center[1] + radius*np.sin(theta)
            z_all = np.append(z_all,z)
            r_all = np.append(r_all,r) 

        # 重複する点を除外する
        is_duplicate = np.zeros(z_all.size,dtype=np.bool_)
        for i in range(r_all.size-1):
            res = abs(z_all[i]-z_all[i+1:])+ abs(r_all[i]-r_all[i+1:])
            is_duplicate[i] = np.any(res < delta_l/100)

        r_all = r_all[~is_duplicate]
        z_all = z_all[~is_duplicate]

        print('num of bound point is ',r_all.size)
        self.__r_bd = r_all
        self.__z_bd = z_all 
        self.__nb = self.z_bd.size

    @property
    def r_bd(self) -> np.ndarray: return self.__r_bd
    @property
    def z_bd(self) -> np.ndarray: return self.__z_bd
    @property
    def nb(self) -> int: return self.__nb


    def internal_grid(self,
            r_grid:np.ndarray,
            z_grid:np.ndarray,
            static:bool=False):
        """
        set internal grid for the vessel
        """
        mask,extent  = self.vessel.detect_grid(r_grid=r_grid, z_grid=z_grid, static=static,isnt_print=True)

        return mask, {"origin":"lower","extent":extent}


    
    def save_point(self,
        name:str,
        is_plot:bool=False,
        fig:plt.Figure | None = None,
        ):
        np.savez(file=name,
                 z_idc=self.z_idc,
                 r_idc=self.r_idc,
                 r_bd=self.r_bd,
                 z_bd=self.z_bd)
        print('inducing points: '+str(self.nI)+' and boundary points: '+str(self.nb)+' are correctly saved at '+name)

        if is_plot:

            self.plot_points(fig=fig,save_name=name)


    def load_point(self,
            r_idc: np.ndarray,
            z_idc: np.ndarray,
            r_bd: np.ndarray,
            z_bd: np.ndarray,
            length_sq_fuction: Callable[[np.ndarray,np.ndarray],np.ndarray],
            factor: float = 1.0,
            is_plot: bool = False,
            fig: plt.Figure | None = None
        ) :  
        """
        set induced point by input existing data

        Parameters
        ----------
        zI: np.ndarray,
        rI: np.ndarray,
        length_sq_fuction: Callable[[float,float],None]
        """
        self.__z_idc, self.__r_idc = z_idc, r_idc
        self.__z_bd, self.__r_bd = z_bd, r_bd
        self.__nI = r_idc.size
        self.__nb = r_bd.size

        self.register_ls_sq_function(length_sq_fuction,factor=factor)
        if is_plot:
            self.plot_points(fig=fig)


    def plot_points(self,
        fig: plt.Figure | None = None,
        save_name: str|None = None,
        ) -> None:
        """
        plot induced points and boundary points

        Parameters
        ----------
        fig: plt.Figure | None = None,
            figure to plot

        """

        if fig is None:    
            if  plot_utils.JOURNAL_MODE ==True:
                fig = plt.figure(figsize=(5,3))
            else:
                fig = plt.figure(figsize=(10,5))
        else:
            fig = fig

        axs = fig.subplots(1,2)
        axs:list[plt.Axes] = np.array(axs).tolist()


        #rmaxとrminを、r_boundの範囲より5%大きくする
        rmax = self.r_bd.max() + 0.05*(self.r_bd.max()-self.r_bd.min())
        rmin = self.r_bd.min() - 0.05*(self.r_bd.max()-self.r_bd.min())

        zmax = self.z_bd.max() + 0.05*(self.z_bd.max()-self.z_bd.min())
        zmin = self.z_bd.min() - 0.05*(self.z_bd.max()-self.z_bd.min())

        for ax in axs:
            ax.set_xlim(rmin,rmax)
            ax.set_ylim(zmin,zmax)
            ax.set_aspect('equal')

                    
        r_plot = np.linspace(self.r_bd.min(),self.r_bd.max(),250)
        z_plot = np.linspace(self.z_bd.min(),self.z_bd.max(),250)
        R,Z = np.meshgrid(r_plot,z_plot)

        mask, im_kwargs = self.internal_grid(r_grid=r_plot,z_grid=z_plot,static=False)

        LS = self.length_scale_func(R,Z)


        plot_utils.contourf_cbar(axs[0],LS*mask,cmap='turbo',vmin=0,**im_kwargs)  

        axs[0].set_title('Length scale distribution')
            
        axs[1].scatter(self.r_idc,self.z_idc,s=1,label='inducing_point')
        title = 'Inducing ponit: '+ str(self.nI)
        if 'r_bd'  in dir(self):
            axs[1].scatter(self.r_bd, self.z_bd,s=1,label='boundary_point')
            title += '\nBoundary ponit: '+ str(self.nb)

        self.vessel.plot(axs[0])
        axs[1].set_title(title)
        axs[1].legend()

        if save_name is not None:
            fig.suptitle(save_name+'.npz')
            fig.savefig(save_name+'.png')
        plt.show()
        
    
    def set_grid_interface(self,
            r_plot: np.ndarray,
            z_plot: np.ndarray,
            z_medium   : np.ndarray | None = None,
            r_medium   : np.ndarray | None = None,
            scale    : float = 1,
            add_bound :bool=False,
        ) :
        """
        Set interface between induced point and grid structure. 
        After this function, you can use convert_grid() to convert into grid structure with r_plot x z_plot.
        """
        
        if not 'r_idc'  in dir(self):
            print('set_induced_point() or create_induced_point() is to be done in advance')
            return
        
        if (len(r_plot.shape) != 1) or (len(z_plot.shape) != 1):
            raise ValueError('r_plot and z_plot should be 1D array')
        
        self.r_plot,self.z_plot = r_plot,z_plot
        ls_min = self.ls_idc.min()*scale

        if z_medium is None:
            z_medium = np.linspace(z_plot.min(),z_plot.max(),int( (z_plot.max() - z_plot.min())/ls_min) + 1)  
        if r_medium is None:
            r_medium = np.linspace(r_plot.min(),r_plot.max(),int( (r_plot.max() - r_plot.min())/ls_min) + 1)

        print( f'medium grid: {r_medium.size} x {z_medium.size}')
        s = scale
        
        Z_medium,R_medium  = np.meshgrid(z_medium, r_medium, indexing='ij')
        lm = self.length_scale_func(R_medium.flatten(), Z_medium.flatten())
        lm = np.nan_to_num(lm,nan=1)

        if add_bound:
            self.add_bound=True

            rIb = np.concatenate([self.r_idc,self.r_bd])
            zIb = np.concatenate([self.z_idc,self.z_bd])

            self.r_idcb,self.z_idcb=rIb,zIb
            lI = self.length_scale(rIb,zIb)
            KII = GibbsKer(x0=rIb, x1=rIb, y0=zIb, y1=zIb, lx0=lI*s, lx1=lI*s, isotropy=True)
            self.KII_inv = np.linalg.inv(KII+1e-5*np.eye(self.nI+self.nb))
            self.KpI = GibbsKer(x0 = R_medium.flatten(),x1 = rIb, y0 = Z_medium.flatten(), y1 =zIb, lx0=lm*s, lx1=lI*s, isotropy=True)        
        else:
            self.add_bound=False
            lI = self.ls_idc
            KII = GibbsKer(x0=self.r_idc, x1=self.r_idc, y0=self.z_idc, y1=self.z_idc, lx0=lI*s, lx1=lI*s, isotropy=True)
            self.KII_inv = np.linalg.inv(KII+1e-5*np.eye(self.nI))
            
            self.KpI = GibbsKer(x0 = R_medium.flatten(),x1 = self.r_idc, y0 = Z_medium.flatten(), y1 =self.z_idc, lx0=lm*s, lx1=lI*s, isotropy=True)
            
        self.r_medium,self.z_medium = r_medium,z_medium
        self.mask_m,self.im_kwargs_m = self.internal_grid(r_grid=r_medium,z_grid=z_medium)

        # grid to grid interface uging the method of kronecker product see(doi: ) 
    
        dr, dz = r_medium[1]-r_medium[0],   z_medium[1]-z_medium[0]

        Kr1r1 = SEKer(x0=r_medium ,x1=r_medium, y0=0., y1=0., lx=dr, ly=1)
        Kz1z1 = SEKer(x0=z_medium ,x1=z_medium, y0=0., y1=0., lx=dz, ly=1)
        
        λ_r1, self.Q_r1 = np.linalg.eigh(Kr1r1)
        λ_z1, self.Q_z1 = np.linalg.eigh(Kz1z1)

        self.mask, self.im_kwargs = self.internal_grid(r_grid=r_plot,z_grid=z_plot)

        self.KrHDr1 = SEKer(x0=r_plot,x1=r_medium, y0=0, y1=0, lx=dr, ly=1)
        self.KzHDz1 = SEKer(x0=z_plot,x1=z_medium, y0=0, y1=0, lx=dz, ly=1)

        self.Λ_z1r1_inv = 1 / np.einsum('i,j->ij',λ_z1,λ_r1)

    
    def convert_grid_media(self,
        fI:np.ndarray,
        boundary:float=0
        ):
        if self.add_bound:
            fI = np.concatenate([fI,boundary*np.ones(self.nb)])
        f1 = self.KpI @ ( self.KII_inv @ fI)
        return f1.reshape(self.mask_m.shape)
    
    
    def convert_grid(self, 
        fI:np.ndarray,
        boundary:float=0,
        ) -> Tuple[np.ndarray,np.ndarray,dict]:
        f1  = self.convert_grid_media(fI,boundary)
        f_HD = self.KzHDz1 @ (self.Q_z1 @ (self.Λ_z1r1_inv * (self.Q_z1.T @ f1 @ self.Q_r1)) @ self.Q_r1.T) @ self.KrHDr1.T
        return f_HD

    def KII_pure_inv(self):
        lI = self.length_scale_func(self.r_idc,self.z_idc)
        KII = GibbsKer(x0=self.r_idc, x1=self.r_idc, y0=self.z_idc, y1=self.z_idc, lx0=lI, lx1=lI, isotropy=True)
        return np.linalg.inv(KII+1e-5*np.eye(self.nI))

    
    def KII_dr(self):
        lI = self.length_scale_fuc(self.r_idc,self.z_idc)
        return  GibbsKer_dx0(self.r_idc,self.r_idc,self.z_idc,self.z_idc,lx0=lI,lx1=lI,isotropy=True)
    
    def KII_dz(self):   
        lI = self.length_scale_func(self.r_idc,self.z_idc)
        return  GibbsKer_dy0(self.r_idc,self.r_idc,self.z_idc,self.z_idc,lx0=lI,lx1=lI,isotropy=True)
    

    
    def set_kernel(self,
            length_scale_factor:float=1,
            is_bound :bool=True ,
            bound_value : float=0,
            bound_sig : float = 0.1,
            is_static_kernel:bool = True,  
            zero_value_index : Optional[npt.NDArray[np.bool_]] = None,
            mean: float= 0,
            out_scale_of_kernel : float = 1,
            eps : float = 1e-6 

        )->Tuple[np.ndarray,np.ndarray]:

        """

        Parameters
        ----------
        length_scale     :,
        is_bound         : Trueのとき境界条件が定められる。,
        bound_value      :,logGPのときは-4から-6がよい
        bound_sig        :,境界における事後分散を指定する。
        is_static_kernel : TrueのときオブジェクトにKf_priとmuf_priが保存される。,
        zero_value_index :  bound_valueと等しい値を持足せたいときのインデックス
        zero_value_sig_factor :  重み(sig_value)を係数
        eps              : 逆行列計算のための微小値

        Reuturns
        ----------
        K_ff_pri: hoge,
        mu_f_pri: hoge,

        """
        lf = length_scale_factor
        lI = lf*self.length_scale_func(self.r_idc,self.z_idc)
        Kii = GibbsKer(x0=self.r_idc     , x1=self.r_idc     , y0=self.z_idc     , y1=self.z_idc     , lx0=lI*lf, lx1=lI*lf, isotropy=True)
        if not is_bound: 
            mu_f_pri = np.zeros_like(self.r_idc)
            Kf_pri = Kii 
        else:
            if (bound_sig < 0) | (bound_sig >= 1):
                raise ValueError('bound_sig must be non-negative')
            rb,zb = self.r_bd,self.z_bd
            factor = 1/ (1-bound_sig**2) 
            
            if zero_value_index is not None:
                index = zero_value_index
                zb,rb = np.concatenate([self.z_idc[index],zb]), np.concatenate([self.r_idc[index],rb])            

            lb = lf*self.length_scale_func(rb,zb)

            KIb = GibbsKer(x0=self.r_idc, x1=rb, y0=self.z_idc, y1=zb, lx0=lI*lf, lx1=lb*lf, isotropy=True)
            Kbb = GibbsKer(x0=rb     , x1=rb, y0=zb     , y1=zb, lx0=lb*lf, lx1=lb*lf, isotropy=True)
            Kernel_bb = factor*Kbb 
            Kernel_bb_inv = np.linalg.inv(Kernel_bb+eps*np.eye(rb.size)) 

            K_bdc = Kii - KIb @  Kernel_bb_inv @ KIb.T
            Kf_pri  = out_scale_of_kernel**2*K_bdc
            mu_f_pri  = mean + KIb @ (Kernel_bb_inv @ (bound_value*np.ones(rb.size)-mean))

            self.temp = KIb
            self.temp2 = Kernel_bb_inv
            self.temp3 = Kernel_bb



            

        if is_static_kernel:
            self.Kf_pri = Kf_pri
            self.muf_pri = mu_f_pri 
            self.kernel_type = 'isotropic kernel'
                    
            self.Kf_pri_property = {
                #'kernel_type': self.kernel_type,
                'is_bound'   : is_bound ,
                #'mean_value' : mean_value,
                'bound_value': bound_value,
                'bound_sig'  : bound_sig, } 

        return Kf_pri,mu_f_pri
    
    def create_obs_matrix_kernel_weighting(self,
        ray  : zray.main.Ray,
        Lnum : int=100
        ) :
        """
        create observation matrix for kernel weighting
        """
        im_shape:tuple = ray.Length.im_shape
        M  = im_shape[0] * im_shape[1]
        

        H = np.zeros((M, self.r_idc.size))

        Rray, Zray = ray.generate_rz(Lnum=Lnum+1)
        dL = ray.Length / float(Lnum)
        
        Zray =0.5*(Zray[:,1:] + Zray[:,:-1])
        Rray =0.5*(Rray[:,1:] + Rray[:,:-1])

        lI = self.length_scale_func(self.r_idc,self.z_idc)

        for i  in tqdm(range(M)):  
            R    = Rray[i,:]
            Z    = Zray[i,:]
            dL2  = dL[i]
            l_ray = self.length_scale_func(R,Z) 
            #Krs =  GibbsKer(x0=R, x1=self.r_idc, y0=Z, y1=self.z_idc, lx0=l_ray*0.5, lx1=lI*0.5,isotropy=True)
            Krs = GibbsKer_isotropy_fast(x0=R, x1=self.r_idc, y0=Z, y1=self.z_idc, l0=l_ray*0.5, l1=lI*0.5)
            #Krs = np.zeros((R.size,self.r_idc.size)) 

            Krs_sum_inv   = 1/Krs.sum(axis=1)

            H[i,:] = np.einsum('i,ij->j', dL2*Krs_sum_inv, Krs ) 
        
        H[H < 1e-5] = 0

        return H
    
    def create_obs_matrix_kernel_interpolation(self,
        ray  : zray.main.Ray,
        Lnum : int=100,
        eps: float = 1e-6,
        ) :
        im_shape:tuple = ray.Length.im_shape
        M  = im_shape[0] * im_shape[1]

        H = np.zeros((M, self.r_idc.size))

        K_ii = GibbsKer(x0=self.r_idc, x1=self.r_idc, y0=self.z_idc, y1=self.z_idc, lx0=self.ls_idc, lx1=self.ls_idc,isotropy=True)
        K_ii_inv = np.linalg.inv(K_ii + eps*np.eye(self.nI))


        Rray, Zray = ray.generate_rz(Lnum=Lnum+1)
        dL = ray.Length / float(Lnum)

        for i  in tqdm(range(M)):  
            R    = Rray[i,:]
            Z    = Zray[i,:]
            dL2  = dL[i]

            Ls_ray = self.length_scale_func(R,Z)


            #K_ray_idc = GibbsKer(x0=R, x1=self.r_idc, y0=Z, y1=self.z_idc, lx0=Ls_ray, lx1=self.ls_idc,isotropy=True)
            K_ray_idc = GibbsKer_isotropy_fast(x0=R, x1=self.r_idc, y0=Z, y1=self.z_idc, l0=Ls_ray, l1=self.ls_idc)
            #K_ray_idc = np.zeros((R.size,self.r_idc.size))
            deltaH = dL2*(K_ray_idc@ K_ii_inv).sum(axis=0)
            H[i,:] = deltaH


        return H
    
    
    def set_unifom_kernel(self,
                          
            length_scale:float=0.1,
            is_bound :bool=True ,
            mean : float=0.,
            bound_value : float=0,
            bound_sig : float = 0.1,
            is_static_kernel:bool = False,  
            out_scale_of_kernel : float = 1,
            zero_value_index : Optional[npt.NDArray[np.bool_]] = None,
            eps : float = 1e-6 

        )->Tuple[np.ndarray,np.ndarray]:

        """
        Parameters
        ----------
        length_scale     :
        is_bound         : Trueのとき境界条件が与えられる。
        mean_value       : 
        bound_value      :
        bound_sig        : 
        is_static_kernel : TrueのときオブジェクトにKf_priとmuf_priが保存される。

        Reuturns
        ----------
        K_ff_pri:
        mu_f_pri:
        """

        ls = length_scale
        Kii = SEKer(x0=self.r_idc, x1=self.r_idc, y0=self.z_idc, y1=self.z_idc, lx=ls, ly=ls)
        
        if not is_bound: 
            mu_f_pri = mean*np.ones_like(self.r_idc)
            Kf_pri = Kii 
        else:
            if (bound_sig < 0) | (bound_sig >= 1):
                raise ValueError('bound_sig must be non-negative')
            rb,zb = self.r_bd,self.z_bd
            factor = 1/ (1-bound_sig**2) 
            
            if zero_value_index is not None:
                index = zero_value_index
                zb,rb = np.concatenate([self.z_idc[index],zb]), np.concatenate([self.r_idc[index],rb])            


            KIb = SEKer(x0=self.r_idc, x1=rb, y0=self.z_idc, y1=zb, lx=ls, ly=ls,)
            Kbb = SEKer(x0=rb     , x1=rb, y0=zb     , y1=zb, lx=ls, ly=ls,)

            Kernel_bb = factor*Kbb 
            Kernel_bb_inv = np.linalg.inv(Kernel_bb+eps*np.eye(rb.size)) 

            K_bdc = Kii - KIb @  Kernel_bb_inv @ KIb.T
            Kf_pri  = out_scale_of_kernel**2*K_bdc
            mu_f_pri  = mean + KIb @ (Kernel_bb_inv @ (bound_value*np.ones(rb.size)-mean))


        if is_static_kernel:
            self.Kf_pri = Kf_pri
            self.muf_pri = mu_f_pri 
            self.kernel_type = 'uniform SE kernel'
                    
            self.Kf_pri_property = {
                #'kernel_type': self.kernel_type,
                'is_bound'   : is_bound ,
                'mean' : mean,
                'bound_value': bound_value,
                'bound_sig'  : bound_sig } 


        return Kf_pri,mu_f_pri

    
    def sampler(self,
        K   : Optional[np.ndarray]=None,
        mu_f: np.ndarray | float = 0.
        ) -> np.ndarray:

        if K is None:
            K = self.Kf_pri
            mu_f = self.muf_pri

        K_hash = hash((K.sum(axis=1)).tobytes())  #type: ignore

        if self.V is None or (self.K_hash != K_hash):
            print('Eigenvalue decomposition is recalculated')
            lam,V = np.linalg.eigh(K) #type: ignore
            lam[lam<1e-5]= 1e-5
            self.V = V
            self.lam = lam
        else:
            self.V = self.V
            self.lam = self.lam
        
        self.K_hash = K_hash 
        
        noise = np.random.randn(self.nI)
        return  mu_f+ self.V @ (np.sqrt(self.lam) *  noise)  
    
    
    
    def plot_mosaic(self,
        f:np.ndarray,
        ax:plt.Axes = None,      
        size :float = 1.0, # type: ignore
        back_ground:float | None =None,
        cbar :bool=True,
        cbar_title: str|None = None,
        is_frame:bool=True,
        vmean: float|None=None,
        **kwargs_scatter,
        )->None:

        if ax is None: ax = plt.gca()

        if 'vmax' in kwargs_scatter:
            vmax = kwargs_scatter['vmax']
        else:
            vmax = np.percentile(f,99)
        if 'vmin' in kwargs_scatter:
            vmin = kwargs_scatter['vmin']
        else:
            vmin = np.percentile(f,1)

    
        if vmean is not None:
            temp = ((vmax - vmean)  > (vmean-vmin))
            tempi = not ((vmax - vmean)  > (vmean-vmin))

            vmax = temp  *vmax + tempi *(2*vmean-vmin)
            vmin = tempi *vmin + temp  *(2*vmean-vmax)
        
        kwargs_scatter['vmax'] = vmax
        kwargs_scatter['vmin'] = vmin

        if back_ground is not None:
            cmap:str = 'viridis'
            alpha =1.0
            if 'cmap' in kwargs_scatter:
                cmap = str(kwargs_scatter['cmap'])
            if 'alpha' in kwargs_scatter:
                alpha = kwargs_scatter['alpha']
            ax.imshow(back_ground*self.mask,cmap=cmap,vmax=vmax,vmin=vmin,alpha=alpha,**self.im_kwargs) # type: ignore
        
        size:np.ndarray = size**2*1e4 *self.ls_sq_idc
        im = ax.scatter(x=self.r_idc,y=self.z_idc,c=f,s=size,**kwargs_scatter)

        if cbar:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right' , size="5%", pad='3%')
            cbar = plt.colorbar(im, cax=cax, orientation='vertical')
            if cbar_title is not None: cbar.set_label(cbar_title) # type: ignore
        
        #ax.set_aspect('equal')
        if is_frame: self.vessel.plot(ax=ax) # type: ignore
                    

class Kernel2D_grid():
    def __init__(self,
        vessel: zray.vessel.AxisymmetricVessel =None,
        r_grid:np.ndarray = None,
        z_grid:np.ndarray = None,
        r_bd:np.ndarray = None,
        z_bd:np.ndarray = None,
        ) -> None:
        
        """
        import dxf file
        from numba import njit

        Parameters
        ----------
        dxf_file : str
            Path of the desired file.
        show_print : bool=True,
            print property of frames
        Note
        ----
        dxf_file is required to have units of (mm).
        """
        self.vessel = vessel    
        self.V = None
        #self.im_shape: Union[Tuple[int,int],None] = None
        if r_grid is not None and z_grid is not None:
            self.set_inducing_point(r_grid,z_grid)

        if r_bd is not None and z_bd is not None:
            self.set_bound(r_bd,z_bd) 


    def set_inducing_point(self,r_grid:np.ndarray,z_grid:np.ndarray) -> None:
        """
        set induced point by input existing data
            
        """
        self.r_grid = r_grid
        self.z_grid = z_grid

        R_grid, Z_grid = np.meshgrid(r_grid,z_grid,indexing='xy')
        self.__idc_shape = R_grid.shape

        self.__r_idc, self.__z_idc = R_grid.flatten(), Z_grid.flatten()
        self.__nI = R_grid.size
        self.unset_bound()

    def set_bound_index(self, index:np.ndarray) -> None:
        """
        set boundary point by input existing data
            
        """
        self.bd_index = index.reshape(self.idc_shape)
        self.__r_bd, self.__z_bd = self.r_idc[index.ravel()], self.z_idc[index.ravel()]
        self.__nb = index.size
        self.boundary = True

    #def set_bound(self,r_bd:np.ndarray,z_bd:np.ndarray) -> None:
    #    """
    #    set boundary point by input existing data
    #        
    #    """
    #    self.__r_bd, self.__z_bd = r_bd, z_bd
    #    self.__nb = r_bd.size
    #    self.boundary = True
    
    def unset_bound(self) -> None:
        """
        unset boundary point
            
        """
        self.bd_index = None
        self.__r_bd, self.__z_bd = np.zeros(0), np.zeros(0)
        self.__nb = 0
        self.boundary = False

    def plot_points(self,ax:plt.Axes = None) -> None:
        """
        plot induced points and boundary points

        Parameters
        ----------
        fig: plt.Figure | None = None,
            figure to plot

        """
        import plot_utils

        if ax is None:
            ax = plt.gca()        

        r_min = self.r_grid.min() - 0.05*(self.r_grid.max()-self.r_grid.min())
        r_max = self.r_grid.max() + 0.05*(self.r_grid.max()-self.r_grid.min())
        z_min = self.z_grid.min() - 0.05*(self.z_grid.max()-self.z_grid.min())
        z_max = self.z_grid.max() + 0.05*(self.z_grid.max()-self.z_grid.min())

        ax.set_xlim(r_min,r_max)
        ax.set_ylim(z_min,z_max)
        ax.set_aspect('equal')

        ax.scatter(self.r_idc,self.z_idc,s=1,label='inducing_point: '+str(self.nI))
        if self.boundary:
            ax.scatter(self.r_bd, self.z_bd,s=1,label='boundary_point: '+str(self.nb))
        
        plt.legend()
        if self.vessel is not None: 
            self.vessel.plot(ax=ax)
        plt.show()

    def set_grid_interface(self,
            r_plot: np.ndarray,
            z_plot: np.ndarray,
        ) :
        """
        Set interface between induced point and grid structure. 
        After this function, you can use convert_grid() to convert into grid structure with r_plot x z_plot.
        """
        
        if not 'r_idc'  in dir(self):
            print('set_induced_point() or create_induced_point() is to be done in advance')
            return
        
        if (len(r_plot.shape) != 1) or (len(z_plot.shape) != 1):
            raise ValueError('r_plot and z_plot should be 1D array')
        
        self.r_plot,self.z_plot = r_plot,z_plot


        r_medium = self.r_grid
        z_medium = self.z_grid



        # grid to grid interface uging the method of kronecker product see(doi: ) 
    
        dr, dz = abs(r_medium[1]-r_medium[0]),   abs(z_medium[1]-z_medium[0])

        self.ls_r_min = dr*0.7
        self.ls_z_min = dz*0.7

        K_ii = SEKer(x0=self.r_idc, x1=self.r_idc, y0=self.z_idc, y1=self.z_idc, lx=  self.ls_r_min, ly= self.ls_z_min)
        
        self.K_ii = K_ii
        a,V = np.linalg.eigh(K_ii) #type: ignore
        a[a<1e-6] = 1e-6
        self.K_ii_inv = V @ np.diag(1/a) @ V.T

        Kr1r1 = SEKer(x0=r_medium ,x1=r_medium, y0=0., y1=0., lx=self.ls_r_min, ly=1)
        Kz1z1 = SEKer(x0=z_medium ,x1=z_medium, y0=0., y1=0., lx=self.ls_z_min, ly=1)
        
        λ_r1, self.Q_r1 = np.linalg.eigh(Kr1r1)
        λ_z1, self.Q_z1 = np.linalg.eigh(Kz1z1)

        if self.vessel is not None:
            self.mask_plt, self.im_kwargs_plt,self.fill_plt = self.internal_grid(r_grid=self.r_plot,z_grid=self.z_plot,static=False)
            self.mask_idc, self.im_kwargs_idc,self.fill_idc = self.internal_grid(r_grid=self.r_idc,z_grid=self.z_idc,static=False)

        else:
            self.im_kwargs_idc = {"origin":"lower","extent":(self.r_idc.min()-0.5*dr,self.r_idc.max()+0.5*dr,self.z_idc.min()-0.5*dz,self.z_idc.max()+0.5*dz)}
            dr2,dz2 = self.r_plot[1]-self.r_plot[0], self.z_plot[1]-self.z_plot[0]
            self.im_kwargs_plt = {"origin":"lower","extent":(self.r_plot.min()-0.5*dr2,self.r_plot.max()+0.5*dr2,self.z_plot.min()-0.5*dz2,self.z_plot.max()+0.5*dz2)}

        self.KrHDr1 = SEKer(x0=r_plot,x1=r_medium, y0=0, y1=0, lx=self.ls_r_min, ly=1)
        self.KzHDz1 = SEKer(x0=z_plot,x1=z_medium, y0=0, y1=0, lx=self.ls_z_min, ly=1)

        self.Λ_z1r1_inv = 1 / np.einsum('i,j->ij',λ_z1,λ_r1)


    def internal_grid(self,
            r_grid:np.ndarray,
            z_grid:np.ndarray,
            static:bool=False):
        """
        set internal grid for the vessel
        """
        mask,extent  = self.vessel.detect_grid(r_grid=r_grid, z_grid=z_grid, static=True,isnt_print=True)

        fill = self.vessel.fill.copy() # 2 for inside, 0 for outside, 1 for boundary

        if not static:
            del self.vessel.fill
            del self.vessel.Is_In
            del self.vessel.Is_Out
            del self.vessel.Is_bound


        return mask, {"origin":"lower","extent":extent},fill

    
    def convert_grid(self, 
        f_idc:np.ndarray,
        ) -> np.ndarray:

        f_idc = f_idc.reshape(self.idc_shape)

        f_plot = self.KzHDz1 @ (self.Q_z1 @ (self.Λ_z1r1_inv * (self.Q_z1.T @ f_idc @ self.Q_r1)) @ self.Q_r1.T) @ self.KrHDr1.T
        return f_plot
    
    def inverse_grid(self,
        f_plot:np.ndarray,
        ) -> np.ndarray:

        f_plot = f_plot.reshape((self.z_plot.size,self.r_plot.size))

        res = self.Q_z1.T@ ((1/self.Λ_z1r1_inv) * (self.Q_z1  @(self.KzHDz1.T @ f_plot @ self.KrHDr1) @ self.Q_r1 )) @ self.Q_r1.T
        return res
    
    def set_uniform_kernel(self,
        ls_r:float=0.1,
        ls_z:float=0.1,
        is_bound :bool=True ,
        mean : float=0.,
        bound_value : float=0,
        bound_sig : float = 0.1,
        static:bool = False,  
        out_scale_of_kernel : float = 1,
        zero_value_index : Optional[npt.NDArray[np.bool_]] = None,
        eps : float = 1e-6 
    )->Tuple[np.ndarray,np.ndarray]:

        """
        Parameters
        ----------
        length_scale     :
        is_bound         : Trueのとき境界条件が与えられる。
        mean_value       :
        bound_value      :
        bound_sig        :
        is_static_kernel : TrueのときオブジェクトにKf_priとmuf_priが保存される。
        """
        ls_r, ls_z = ls_r, ls_z
        Kii = SEKer(x0=self.r_idc, x1=self.r_idc, y0=self.z_idc, y1=self.z_idc, lx=ls_r, ly=ls_z)



        
        if not is_bound: 
            mu_f_pri = mean*np.ones_like(self.r_idc)
            Kf_pri = Kii 
        else:
            if (bound_sig < 0) | (bound_sig >= 1):
                raise ValueError('bound_sig must be non-negative')
            rb,zb = self.r_bd,self.z_bd
            factor = 1/ (1-bound_sig**2) 
            
            if zero_value_index is not None:
                index = zero_value_index
                zb,rb = np.concatenate([self.z_idc[index],zb]), np.concatenate([self.r_idc[index],rb])            


            KIb = SEKer(x0=self.r_idc, x1=rb, y0=self.z_idc, y1=zb, lx=ls_r, ly=ls_z,)
            Kbb = SEKer(x0=rb     , x1=rb, y0=zb     , y1=zb, lx=ls_r, ly=ls_z,)

            Kernel_bb = factor*Kbb 
            Kernel_bb_inv = np.linalg.inv(Kernel_bb+eps*np.eye(rb.size)) 

            K_bdc = Kii - KIb @  Kernel_bb_inv @ KIb.T
            Kf_pri  = out_scale_of_kernel**2*K_bdc
            mu_f_pri  = mean + KIb @ (Kernel_bb_inv @ (bound_value*np.ones(rb.size)-mean))


        if static:
            self.Kf_pri = Kf_pri
            self.muf_pri = mu_f_pri 
            self.kernel_type = 'uniform SE kernel'
                    
            self.Kf_pri_property = {
                #'kernel_type': self.kernel_type,
                'length_scale_r': ls_r,
                'length_scale_z': ls_z,
                'is_bound'   : is_bound ,
                'mean' : mean,
                'bound_value': bound_value,
                'bound_sig'  : bound_sig } 


        return Kf_pri,mu_f_pri

    def sampler(self,
        K   : Optional[np.ndarray]=None,
        mu_f: np.ndarray | float = 0.
        ) -> np.ndarray:

        if K is None:
            K = self.Kf_pri
            mu_f = self.muf_pri

        K_hash = hash((K.sum(axis=1)).tobytes())  #type: ignore

        if self.V is None or (self.K_hash != K_hash):
            print('Eigenvalue decomposition is recalculated')
            lam,V = np.linalg.eigh(K) #type: ignore
            lam[lam<1e-5]= 1e-5
            self.V = V
            self.lam = lam
        else:
            self.V = self.V
            self.lam = self.lam
        
        self.K_hash = K_hash 
        
        noise = np.random.randn(self.nI)
        return  mu_f+ self.V @ (np.sqrt(self.lam) *  noise)  
    
    
    @property
    def r_idc(self) -> np.ndarray: return self.__r_idc
    @property
    def z_idc(self) -> np.ndarray: return self.__z_idc
    @property
    def nI(self) -> int: return self.__nI
    @property
    def r_bd(self) -> np.ndarray: return self.__r_bd
    @property
    def z_bd(self) -> np.ndarray: return self.__z_bd
    @property
    def nb(self) -> int: return self.__nb
    @property
    def idc_shape(self) -> Tuple[int,int]: return self.__idc_shape

    

@njit
def d2min(x,y,xs,ys):
    x_tau2 = (x- xs)**2
    y_tau2 = (y- ys)**2
    d2_min = np.min(x_tau2 + y_tau2)
    return d2_min

def SEKer(
    x0 : np.ndarray,
    x1 : np.ndarray,
    y0 : np.ndarray |float,
    y1 : np.ndarray |float,
    lx : float,
    ly : float,
    ) -> np.ndarray:

    X = np.meshgrid(x0,x1,indexing='ij')
    Y = np.meshgrid(y0,y1,indexing='ij')
    return np.exp(- 0.5*( ((X[0]-X[1])/abs(lx))**2 + ((Y[0]-Y[1])/abs(ly))**2) )

def GibbsKer(
    x0 : np.ndarray,
    x1 : np.ndarray,
    y0 : np.ndarray,
    y1 : np.ndarray,
    lx0: np.ndarray,
    lx1: np.ndarray,
    ly0: np.ndarray | bool  = False,
    ly1: np.ndarray | bool  = False,
    isotropy: bool = False
    ) -> np.ndarray:  

    X  = np.meshgrid(x0,x1,indexing='ij')
    Y  = np.meshgrid(y0,y1,indexing='ij')
    Lx = np.meshgrid(lx0,lx1,indexing='ij')
    Lxsq = Lx[0]**2+Lx[1]**2 

    if isotropy:
        return 2*Lx[0]*Lx[1]/Lxsq *np.exp( -   ((X[0]-X[1])**2  +(Y[0]-Y[1])**2 )/ Lxsq )

    else:
        Ly = np.meshgrid(ly0,ly1,indexing='ij')
        Lysq = Ly[0]**2+Ly[1]**2 
        return np.sqrt(2*Lx[0]*Lx[1]/Lxsq) *np.sqrt(2*Ly[0]*Ly[1]/Lysq) *np.exp( -(X[0]-X[1])**2 / Lxsq  - (Y[0]-Y[1])**2 / Lysq )# type: ignore


def Kernel_SE_2dim(x1,x2,y1,y2,lx,ly):
    X1,X2 = np.meshgrid(x1,x2,indexing='ij')
    Y1,Y2 = np.meshgrid(y1,y2,indexing='ij')
    K = np.exp(-0.5*(X1-X2)**2/lx**2 -0.5*(Y1-Y2)**2/ly**2)
    return K

def Kernel_SE_2dim_dx1(x1,x2,y1,y2,lx,ly):
    X1,X2 = np.meshgrid(x1,x2,indexing='ij')
    Y1,Y2 = np.meshgrid(y1,y2,indexing='ij')
    K = Kernel_SE_2dim(x1,x2,y1,y2,lx,ly)
    return  -(X1-X2)/lx**2*K

def GibbsKer_dx0(
    x0 : np.ndarray,
    x1 : np.ndarray,
    y0 : np.ndarray,
    y1 : np.ndarray,
    lx0: np.ndarray,
    lx1: np.ndarray,
    ly0: np.ndarray | bool  = False,
    ly1: np.ndarray | bool  = False,
    isotropy: bool = False
    ) -> np.ndarray:  

    X  = np.meshgrid(x0,x1,indexing='ij')
    Y  = np.meshgrid(y0,y1,indexing='ij')
    Lx = np.meshgrid(lx0,lx1,indexing='ij')
    Lxsq = Lx[0]**2+Lx[1]**2

    if isotropy:
        return -4*(X[0]-X[1])*Lx[0]*Lx[1]/Lxsq**2 *np.exp( -   ((X[0]-X[1])**2  +(Y[0]-Y[1])**2 )/ Lxsq )
    
    else:
        Ly = np.meshgrid(ly0,ly1,indexing='ij')
        Lysq = Ly[0]**2+Ly[1]**2 
        return -2*np.sqrt(2*Lx[0]*Lx[1]/Lxsq) /Lxsq *np.sqrt(2*Ly[0]*Ly[1]/Lysq) *np.exp( -(X[0]-X[1])**2 / Lxsq  - (Y[0]-Y[1])**2 / Lysq )
    
def GibbsKer_dy0(
    x0 : np.ndarray,
    x1 : np.ndarray,
    y0 : np.ndarray,
    y1 : np.ndarray,
    lx0: np.ndarray,
    lx1: np.ndarray,
    ly0: np.ndarray | bool  = False,
    ly1: np.ndarray | bool  = False,
    isotropy: bool = False
    ) -> np.ndarray:  

    X  = np.meshgrid(x0,x1,indexing='ij')
    Y  = np.meshgrid(y0,y1,indexing='ij')
    Lx = np.meshgrid(lx0,lx1,indexing='ij')
    Lxsq = Lx[0]**2+Lx[1]**2
    
    if isotropy:
        return -4*(Y[0]-Y[1])*Lx[0]*Lx[1]/Lxsq**2 *np.exp( -   ((X[0]-X[1])**2  +(Y[0]-Y[1])**2 )/ Lxsq )
    
    else:
        Ly = np.meshgrid(ly0,ly1,indexing='ij')
        Lysq = Ly[0]**2+Ly[1]**2 
        return -2*np.sqrt(2*Lx[0]*Lx[1]/Lxsq) /Lxsq *np.sqrt(2*Ly[0]*Ly[1]/Lysq) *np.exp( -(X[0]-X[1])**2 / Lxsq  - (Y[0]-Y[1])**2 / Lysq )
    
    

@njit
def GibbsKer_isotropy_fast(
    x0 : np.ndarray,
    x1 : np.ndarray,
    y0 : np.ndarray,
    y1 : np.ndarray,
    l0: np.ndarray,
    l1: np.ndarray,
    ) -> np.ndarray:  
    
    n0 = x0.size
    n1 = x1.size
    K = np.empty((n0, n1))
    
    for i in range(n0):
        for j in range(n1):
            lsq = l0[i]**2 + l1[j]**2
            dist_sq = (x0[i] - x1[j])**2 + (y0[i] - y1[j])**2
            K[i, j] = 2 * l0[i] * l1[j] / lsq * np.exp(-dist_sq / lsq)
            
    return K