import numpy as np
from typing import Callable, Optional, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import mpl_toolkits.axes_grid1
import numpy.typing as npt
import os,sys
import scipy.sparse as sparse


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
            lI = self.length_scale_func(rIb,zIb)
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
        H, _ = self.create_obs_and_dcos_kernel_weighting(ray=ray, Lnum=Lnum)
        return H

    @staticmethod
    def _ray_midpoints_xyz_and_dL(
        ray: zray.main.Ray,
        *,
        Lnum: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if Lnum <= 0:
            raise ValueError("Lnum must be positive")
        Xray, Yray, Zray = ray.generate_xyz(Lnum=Lnum + 1)
        Xray = np.asarray(Xray, dtype=float)
        Yray = np.asarray(Yray, dtype=float)
        Zray = np.asarray(Zray, dtype=float)
        if Xray.shape != Yray.shape or Xray.shape != Zray.shape:
            raise ValueError("ray.generate_xyz returned inconsistent shapes")
        Xm = 0.5 * (Xray[:, 1:] + Xray[:, :-1])
        Ym = 0.5 * (Yray[:, 1:] + Yray[:, :-1])
        Zm = 0.5 * (Zray[:, 1:] + Zray[:, :-1])
        dL = np.asarray(ray.Length, dtype=float).reshape(-1) / float(Lnum)
        return Xm, Ym, Zm, dL

    @staticmethod
    def _toroidal_dcos_midpoints_from_xyz(
        x_mid: np.ndarray,
        y_mid: np.ndarray,
        ray_dir_xyz: np.ndarray,
        *,
        eps_r: float = 1e-10,
    ) -> np.ndarray:
        x_mid = np.asarray(x_mid, dtype=float)
        y_mid = np.asarray(y_mid, dtype=float)
        ray_dir_xyz = np.asarray(ray_dir_xyz, dtype=float)
        if x_mid.shape != y_mid.shape:
            raise ValueError("x_mid and y_mid must have the same shape")
        if ray_dir_xyz.ndim != 2 or ray_dir_xyz.shape[1] != 3:
            raise ValueError("ray_dir_xyz must have shape (M, 3)")
        if ray_dir_xyz.shape[0] != x_mid.shape[0]:
            raise ValueError("ray_dir_xyz row count must match midpoint rows")

        r_mid = np.sqrt(x_mid**2 + y_mid**2)
        valid = np.isfinite(r_mid) & np.isfinite(x_mid) & np.isfinite(y_mid) & (r_mid >= eps_r)

        ephi_x = np.zeros_like(x_mid, dtype=float)
        ephi_y = np.zeros_like(y_mid, dtype=float)
        ephi_x[valid] = -y_mid[valid] / r_mid[valid]
        ephi_y[valid] = +x_mid[valid] / r_mid[valid]

        dcos = ray_dir_xyz[:, [0]] * ephi_x + ray_dir_xyz[:, [1]] * ephi_y
        dcos = np.asarray(np.clip(dcos, -1.0, 1.0), dtype=float)
        dcos[~valid] = 0.0
        return dcos

    def create_obs_and_dcos_kernel_weighting(
        self,
        ray: zray.main.Ray,
        *,
        Lnum: int = 100,
        eps_r: float = 1e-10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create kernel-weighted observation matrix and directional-cosine matrix.

        ``Dcos`` corresponds to the directional cosine to the local toroidal basis
        vector and is the practical counterpart of the CIS matrix ``Theta``.
        """
        im_shape: tuple = ray.Length.im_shape
        M = im_shape[0] * im_shape[1]
        H = np.zeros((M, self.r_idc.size), dtype=float)
        Dcos = np.zeros((M, self.r_idc.size), dtype=float)

        Xmid, Ymid, Zmid, dL = self._ray_midpoints_xyz_and_dL(ray=ray, Lnum=Lnum)
        Rmid = np.sqrt(Xmid**2 + Ymid**2)
        ray_dcos_mid = self._toroidal_dcos_midpoints_from_xyz(
            x_mid=Xmid,
            y_mid=Ymid,
            ray_dir_xyz=np.asarray(ray.Direction_xyz, dtype=float),
            eps_r=eps_r,
        )
        lI = self.length_scale_func(self.r_idc, self.z_idc)

        for i in tqdm(range(M)):
            R = Rmid[i, :]
            Z = Zmid[i, :]
            dL_i = float(dL[i])
            l_ray = self.length_scale_func(R, Z)
            Krs = GibbsKer_isotropy_fast(x0=R, x1=self.r_idc, y0=Z, y1=self.z_idc, l0=l_ray * 0.5, l1=lI * 0.5)

            Krs_sum = np.asarray(Krs.sum(axis=1), dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                Krs_sum_inv = np.divide(1.0, Krs_sum, out=np.zeros_like(Krs_sum, dtype=float), where=Krs_sum > 0)

            seg_weights = (dL_i * Krs_sum_inv)[:, None] * Krs
            H_row = np.asarray(seg_weights.sum(axis=0), dtype=float)
            D_num = np.asarray((seg_weights * ray_dcos_mid[i, :][:, None]).sum(axis=0), dtype=float)
            D_row = np.divide(D_num, H_row, out=np.zeros_like(D_num), where=H_row > 0)

            H[i, :] = H_row
            Dcos[i, :] = np.clip(D_row, -1.0, 1.0)

        H[H < 1e-5] = 0.0
        Dcos[H <= 0.0] = 0.0
        return H, Dcos

    def create_dcos_matrix_kernel_weighting(
        self,
        ray: zray.main.Ray,
        *,
        Lnum: int = 100,
        H: np.ndarray | None = None,
        eps_r: float = 1e-10,
    ) -> np.ndarray:
        """Create only the directional-cosine matrix for CIS.

        If ``H`` is provided, it is currently used only for compatibility; the
        matrix is recomputed internally to keep the weighting rule identical.
        """
        _, Dcos = self.create_obs_and_dcos_kernel_weighting(ray=ray, Lnum=Lnum, eps_r=eps_r)
        if H is not None and np.asarray(H).shape != Dcos.shape:
            raise ValueError(f"Provided H shape {np.asarray(H).shape} does not match Dcos shape {Dcos.shape}")
        return Dcos
    
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


class Kernel2D_scatter_grid(Kernel2D_scatter):
    """Scatter-kernel variant with helpers for grid-derived inducing points and grid-binned observation matrices.

    This class is intended to cover the workflow prototyped in old grid-based notebooks:
    1. define a regular (r, z) lattice and vessel fill labels,
    2. pick inducing points from inside cells and boundary points from fill/domain edges,
    3. build a grid-binned observation matrix by ray midpoint binning.
    """

    @staticmethod
    def constant_length_scale_sq_function(
        *,
        length_scale: float | None = None,
        length_scale_sq: float | None = None,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if (length_scale is None) == (length_scale_sq is None):
            raise ValueError("Specify exactly one of length_scale or length_scale_sq")
        if length_scale is not None:
            if length_scale <= 0:
                raise ValueError("length_scale must be positive")
            value = float(length_scale) ** 2
        else:
            if length_scale_sq is None or length_scale_sq <= 0:
                raise ValueError("length_scale_sq must be positive")
            value = float(length_scale_sq)

        def _f(r: np.ndarray, z: np.ndarray) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            z = np.asarray(z, dtype=float)
            shape = np.broadcast(r, z).shape
            return np.full(shape, value, dtype=float)

        return _f

    @staticmethod
    def _nearest_index_monotonic(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
        grid = np.asarray(grid, dtype=float)
        values = np.asarray(values, dtype=float)
        if grid.ndim != 1:
            raise ValueError("grid must be 1D")
        if grid.size < 2:
            return np.zeros(values.shape, dtype=np.int64)
        if np.any(np.diff(grid) < 0):
            raise ValueError("grid must be monotonically nondecreasing")

        idx = np.searchsorted(grid, values, side="left")
        idx = np.clip(idx, 0, grid.size - 1)
        left = np.clip(idx - 1, 0, grid.size - 1)
        right = idx
        choose_left = np.abs(values - grid[left]) <= np.abs(grid[right] - values)
        return np.where(choose_left, left, right).astype(np.int64)

    @staticmethod
    def _normalize_ray_samples(
        Rray: np.ndarray,
        Zray: np.ndarray,
        *,
        M: int,
        sample_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        Rray = np.asarray(Rray, dtype=float)
        Zray = np.asarray(Zray, dtype=float)
        if Rray.shape != Zray.shape:
            raise ValueError("Rray and Zray must have the same shape")

        if Rray.ndim == 2:
            if Rray.shape == (M, sample_count):
                return Rray, Zray
            if Rray.shape == (sample_count, M):
                return Rray.T, Zray.T
            raise ValueError(f"Unsupported 2D ray sample shape: {Rray.shape}")

        if Rray.ndim == 3:
            if Rray.shape[0] == sample_count and np.prod(Rray.shape[1:]) == M:
                return Rray.reshape(sample_count, M).T, Zray.reshape(sample_count, M).T
            if Rray.shape[-1] == sample_count and np.prod(Rray.shape[:-1]) == M:
                return Rray.reshape(M, sample_count), Zray.reshape(M, sample_count)
            raise ValueError(f"Unsupported 3D ray sample shape: {Rray.shape}")

        raise ValueError(f"Unsupported ray sample ndim: {Rray.ndim}")

    def set_inducing_point_from_grid_fill(
        self,
        r_grid: np.ndarray,
        z_grid: np.ndarray,
        fill: np.ndarray,
        *,
        length_sq_fuction: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        length_scale: float | None = None,
        length_scale_sq: float | None = None,
        factor: float = 1.0,
        inside_value: int = 2,
        boundary_value: int = 1,
        include_fill_boundary: bool = True,
        include_domain_edge_inside_boundary: bool = True,
        deduplicate_boundary: bool = True,
        is_plot: bool = False,
        fig: plt.Figure | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build inducing/boundary points from a fill-labeled grid and load them into the scatter kernel."""

        r_grid = np.asarray(r_grid, dtype=float)
        z_grid = np.asarray(z_grid, dtype=float)
        fill = np.asarray(fill)
        if fill.shape != (z_grid.size, r_grid.size):
            raise ValueError("fill shape must be (len(z_grid), len(r_grid))")

        inside_mask = fill == inside_value
        boundary_mask = fill == boundary_value
        Z_grid, R_grid = np.meshgrid(z_grid, r_grid, indexing="ij")

        r_idc = R_grid[inside_mask]
        z_idc = Z_grid[inside_mask]

        r_bd_list: list[np.ndarray] = []
        z_bd_list: list[np.ndarray] = []
        if include_fill_boundary:
            r_bd_list.append(R_grid[boundary_mask])
            z_bd_list.append(Z_grid[boundary_mask])

        if include_domain_edge_inside_boundary and r_idc.size > 0:
            edge_inside = (
                (z_idc == z_grid.min())
                | (z_idc == z_grid.max())
                | (r_idc == r_grid.min())
                | (r_idc == r_grid.max())
            )
            r_bd_list.append(r_idc[edge_inside])
            z_bd_list.append(z_idc[edge_inside])

        if r_bd_list:
            r_bd = np.concatenate(r_bd_list)
            z_bd = np.concatenate(z_bd_list)
        else:
            r_bd = np.zeros(0, dtype=float)
            z_bd = np.zeros(0, dtype=float)

        if deduplicate_boundary and r_bd.size > 0:
            pts = np.column_stack([r_bd, z_bd])
            _, uniq_idx = np.unique(pts, axis=0, return_index=True)
            uniq_idx = np.sort(uniq_idx)
            r_bd = r_bd[uniq_idx]
            z_bd = z_bd[uniq_idx]

        if length_sq_fuction is None:
            length_sq_fuction = self.constant_length_scale_sq_function(
                length_scale=length_scale,
                length_scale_sq=length_scale_sq,
            )

        self.load_point(
            r_idc=r_idc,
            z_idc=z_idc,
            r_bd=r_bd,
            z_bd=z_bd,
            length_sq_fuction=length_sq_fuction,
            factor=factor,
            is_plot=is_plot,
            fig=fig,
        )
        return r_idc, z_idc, r_bd, z_bd

    def set_uniform_kernel(self, *, length_scale: float, **kwargs):
        """Alias for legacy ``set_unifom_kernel`` with a corrected method name."""
        return self.set_unifom_kernel(length_scale=length_scale, **kwargs)

    def create_obs_matrix_grid_binning(
        self,
        ray: zray.Ray,
        *,
        r_grid: np.ndarray,
        z_grid: np.ndarray,
        sample_count: int = 400,
        column_mask: npt.NDArray[np.bool_] | None = None,
        sparse_output: bool = True,
        return_grid4d: bool = False,
        chunk_size: int = 2048,
        show_progress: bool = True,
    ) -> np.ndarray | sparse.csr_matrix | tuple[np.ndarray | sparse.csr_matrix, np.ndarray]:
        """Create a grid-binned observation matrix by ray midpoint binning.

        This reproduces the intent of the old notebook `H_matrix_grid` loop while
        supporting sparse accumulation and optional column masking (e.g. inside cells).
        """

        r_grid = np.asarray(r_grid, dtype=float)
        z_grid = np.asarray(z_grid, dtype=float)
        if r_grid.ndim != 1 or z_grid.ndim != 1:
            raise ValueError("r_grid and z_grid must be 1D arrays")
        if sample_count < 2:
            raise ValueError("sample_count must be >= 2")

        length_arr = np.asarray(ray.Length, dtype=float)
        im_shape = getattr(ray.Length, "im_shape", None)
        M = int(length_arr.size)
        nseg = sample_count - 1
        dL_flat = length_arr.reshape(-1) / float(nseg)

        Rray_raw, Zray_raw = ray.generate_rz(Lnum=sample_count)
        Rray, Zray = self._normalize_ray_samples(Rray_raw, Zray_raw, M=M, sample_count=sample_count)
        Rmid = 0.5 * (Rray[:, 1:] + Rray[:, :-1])
        Zmid = 0.5 * (Zray[:, 1:] + Zray[:, :-1])

        nz, nr = z_grid.size, r_grid.size
        ncols_all = nz * nr

        remap_cols = None
        if column_mask is not None:
            column_mask = np.asarray(column_mask, dtype=bool)
            if column_mask.shape != (nz, nr):
                raise ValueError(f"column_mask shape must be {(nz, nr)}")
            keep_cols = np.flatnonzero(column_mask.ravel())
            remap_cols = np.full(ncols_all, -1, dtype=np.int64)
            remap_cols[keep_cols] = np.arange(keep_cols.size, dtype=np.int64)
            ncols = int(keep_cols.size)
        else:
            ncols = ncols_all

        if return_grid4d:
            if sparse_output:
                raise ValueError("return_grid4d=True requires sparse_output=False")
            if column_mask is not None:
                raise ValueError("return_grid4d=True requires column_mask=None")
            if im_shape is None:
                raise ValueError("return_grid4d=True requires ray.Length.im_shape")

        H_dense = None if sparse_output else np.zeros((M, ncols), dtype=float)
        csr_chunks: list[sparse.csr_matrix] = []
        iterator = range(0, M, chunk_size)
        if show_progress:
            iterator = tqdm(iterator, total=(M + chunk_size - 1) // chunk_size)

        for start in iterator:
            end = min(M, start + chunk_size)
            mb = end - start

            Rb = Rmid[start:end]
            Zb = Zmid[start:end]
            valid = np.isfinite(Rb) & np.isfinite(Zb)

            zi = self._nearest_index_monotonic(z_grid, Zb)
            ri = self._nearest_index_monotonic(r_grid, Rb)
            col_all = zi * nr + ri
            if remap_cols is not None:
                col = remap_cols[col_all]
                valid &= col >= 0
            else:
                col = col_all

            rows_local = np.repeat(np.arange(mb, dtype=np.int64), nseg)
            cols_local = col.reshape(-1)
            valid_flat = valid.reshape(-1)
            rows_local = rows_local[valid_flat]
            cols_local = cols_local[valid_flat]
            data = np.repeat(dL_flat[start:end], nseg)[valid_flat]
            H_chunk = sparse.csr_matrix((data, (rows_local, cols_local)), shape=(mb, ncols))

            if sparse_output:
                csr_chunks.append(H_chunk)
            else:
                H_dense[start:end, :] = H_chunk.toarray()

        H_flat: np.ndarray | sparse.csr_matrix
        if sparse_output:
            H_flat = sparse.vstack(csr_chunks, format="csr") if csr_chunks else sparse.csr_matrix((M, ncols))
        else:
            H_flat = H_dense if H_dense is not None else np.zeros((M, ncols), dtype=float)

        if not return_grid4d:
            return H_flat

        if im_shape is None:
            raise RuntimeError("im_shape unexpectedly missing")
        ny, nx = map(int, im_shape)
        H_grid4d = np.asarray(H_flat).reshape(ny, nx, nz, nr)
        return H_flat, H_grid4d

    def create_obs_and_dcos_grid_binning(
        self,
        ray: zray.Ray,
        *,
        r_grid: np.ndarray,
        z_grid: np.ndarray,
        sample_count: int = 400,
        column_mask: npt.NDArray[np.bool_] | None = None,
        sparse_output: bool = False,
        chunk_size: int = 2048,
        show_progress: bool = True,
        eps_r: float = 1e-10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create grid-binned ``H`` and ``Dcos`` (CIS directional-cosine) matrices.

        Initial implementation prioritizes correctness and uses dense output.
        """
        if sparse_output:
            raise NotImplementedError("sparse_output=True is not implemented for create_obs_and_dcos_grid_binning")

        H = np.asarray(
            self.create_obs_matrix_grid_binning(
                ray,
                r_grid=r_grid,
                z_grid=z_grid,
                sample_count=sample_count,
                column_mask=column_mask,
                sparse_output=False,
                return_grid4d=False,
                chunk_size=chunk_size,
                show_progress=show_progress,
            ),
            dtype=float,
        )

        r_grid = np.asarray(r_grid, dtype=float)
        z_grid = np.asarray(z_grid, dtype=float)
        length_arr = np.asarray(ray.Length, dtype=float)
        M = int(length_arr.size)
        nseg = sample_count - 1
        if nseg <= 0:
            raise ValueError("sample_count must be >= 2")

        Xray_raw, Yray_raw, _ = ray.generate_xyz(Lnum=sample_count)
        Xray, Yray = self._normalize_ray_samples(Xray_raw, Yray_raw, M=M, sample_count=sample_count)
        Xmid = 0.5 * (Xray[:, 1:] + Xray[:, :-1])
        Ymid = 0.5 * (Yray[:, 1:] + Yray[:, :-1])
        dcos_mid = self._toroidal_dcos_midpoints_from_xyz(
            x_mid=Xmid,
            y_mid=Ymid,
            ray_dir_xyz=np.asarray(ray.Direction_xyz, dtype=float),
            eps_r=eps_r,
        )

        Rray_raw, Zray_raw = ray.generate_rz(Lnum=sample_count)
        Rray, Zray = self._normalize_ray_samples(Rray_raw, Zray_raw, M=M, sample_count=sample_count)
        Rmid = 0.5 * (Rray[:, 1:] + Rray[:, :-1])
        Zmid = 0.5 * (Zray[:, 1:] + Zray[:, :-1])

        nz, nr = z_grid.size, r_grid.size
        ncols_all = nz * nr
        remap_cols = None
        if column_mask is not None:
            column_mask = np.asarray(column_mask, dtype=bool)
            if column_mask.shape != (nz, nr):
                raise ValueError(f"column_mask shape must be {(nz, nr)}")
            keep_cols = np.flatnonzero(column_mask.ravel())
            remap_cols = np.full(ncols_all, -1, dtype=np.int64)
            remap_cols[keep_cols] = np.arange(keep_cols.size, dtype=np.int64)
            ncols = int(keep_cols.size)
        else:
            ncols = ncols_all

        if H.shape != (M, ncols):
            raise RuntimeError(f"Unexpected H shape {H.shape}, expected {(M, ncols)}")

        dL_flat = length_arr.reshape(-1) / float(nseg)
        Dnum = np.zeros_like(H, dtype=float)
        iterator = range(0, M, chunk_size)
        if show_progress:
            iterator = tqdm(iterator, total=(M + chunk_size - 1) // chunk_size)

        for start in iterator:
            end = min(M, start + chunk_size)
            mb = end - start

            Rb = Rmid[start:end]
            Zb = Zmid[start:end]
            valid = np.isfinite(Rb) & np.isfinite(Zb)

            zi = self._nearest_index_monotonic(z_grid, Zb)
            ri = self._nearest_index_monotonic(r_grid, Rb)
            col_all = zi * nr + ri
            if remap_cols is not None:
                col = remap_cols[col_all]
                valid &= col >= 0
            else:
                col = col_all

            rows_local = np.repeat(np.arange(mb, dtype=np.int64), nseg)
            cols_local = col.reshape(-1)
            valid_flat = valid.reshape(-1)
            rows_local = rows_local[valid_flat]
            cols_local = cols_local[valid_flat]
            if rows_local.size == 0:
                continue

            vals_dcos = dcos_mid[start:end].reshape(-1)[valid_flat]
            weights = np.repeat(dL_flat[start:end], nseg)[valid_flat] * vals_dcos
            chunk = sparse.csr_matrix((weights, (rows_local, cols_local)), shape=(mb, ncols))
            Dnum[start:end, :] = chunk.toarray()

        Dcos = np.divide(Dnum, H, out=np.zeros_like(Dnum), where=H > 0)
        Dcos = np.clip(Dcos, -1.0, 1.0)
        Dcos[H <= 0.0] = 0.0
        return H, Dcos

    def create_dcos_matrix_grid_binning(self, *args, **kwargs) -> np.ndarray:
        _, Dcos = self.create_obs_and_dcos_grid_binning(*args, **kwargs)
        return Dcos
                    

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
