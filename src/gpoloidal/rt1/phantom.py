from . import mag
import numpy as np
from functools import partial

n0 = 2#25.99e16*0.8/2
a  = 1.348
b  = 0.5
rmax = 0.4577



def gaussian(r,z,n0=n0,a=a,b=b,rmax=rmax,separatrix=True):
    psi = mag.psi(r,z,separatrix=separatrix)
    br, bz = mag.bvec(r,z,separatrix=separatrix)
    b_abs = np.sqrt(br**2+bz**2)
    psi_rmax = mag.psi(rmax,0,separatrix=separatrix)
    psi0 = mag.psi(1,0,separatrix=separatrix)
    b0 = mag.b0(r,z,separatrix=separatrix)
    return n0 * np.exp(-a*(psi-psi_rmax)**2/psi0**2)*(b_abs/b0)**(-b) 

def Length_scale_sq(r,z):
    return 0.0002/(gaussian(r,z)+ 0.05)

def Length_scale(r,z):
    return np.sqrt( Length_scale_sq(r,z))



psi_sep = -0.006376568930277712
def sep_factor(r,z):
    psi = mag.psi(r,z,separatrix=True)
    return  1/(1+np.exp(+1000*(psi-psi_sep)))


def gaussian(r,z,n0=n0,a=a,b=b,rmax=rmax,separatrix=True):
    psi = mag.psi(r,z,separatrix=separatrix)
    br, bz = mag.bvec(r,z,separatrix=separatrix)
    b_abs = np.sqrt(br**2+bz**2)
    psi_rmax = mag.psi(rmax,0,separatrix=separatrix)
    psi0 = mag.psi(1,0,separatrix=separatrix)
    b0 = mag.b0(r,z,separatrix=separatrix)
    return n0 * np.exp(-a*(psi-psi_rmax)**2/psi0**2)*(b_abs/b0)**(-b) 

def func_ring(r,z,n0=n0,a=a,b=b,rmax=rmax,radius=0.5,separatrix=True):
    psi = mag.psi(r,z,separatrix)
    br, bz = mag.bvec(r,z,separatrix)
    b_abs = np.sqrt(br**2+bz**2)
    psi_rmax = mag.psi(rmax,0,separatrix)
    psi0 = mag.psi(1,0,separatrix)
    b0 = mag.b0(r,z,separatrix)
    b2 = b_abs/b0
    if separatrix:
        rs,zs = r.flatten()[np.argmin(b_abs)],z.flatten()[np.argmin(b_abs)]
        fac = (1-np.exp(-50*((r-rs)**2+(z-zs)**2)))
    else:
        fac = 1


    #f_gaussian = fac*n0 * np.exp(- (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-b/b2)**2)-radius)**2*1/a**2)*(1-np.exp(-100*(r-1)**2))
    f_cauchy =  fac*n0 * a**2/( (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-b/b2)**2)-radius)**2+a**2)*(1-np.exp(-100*(r-1)**2))
    return f_cauchy
    #return n0 *  (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-1/b)**2)-0.5)*a / (a +   (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-1/b)**2)-0.5)**2)*(1-np.exp(-100*(r-1)**2))


#f_ring2_HD =func_ring(r=R_grid,z=Z_grid,n0=1,a=0.06,b=0.8,rmax=0.58,radius=0.42)
#f_ring2 =   func_ring(r=rI,z=zI,    n0=1,a=0.06,b=0.8,rmax=0.58,radius=0.42)


def get_phantom_funtion(name:str):
    if name  in ['Hollow','hollow']:
        return partial(func_ring,n0=1,a=0.06,b=0.8,rmax=0.58,radius=0.42) 
    
    elif name in ['single','Single','Single Gaussian']:
                

        def func(r,z):
            f =  gaussian(r,z,n0=1,a=6.3,b=1.0,rmax=0.53)*sep_factor(r,z)

            f[z>0.48] = 0
             
            return f    
        
        return func
    
    elif name in ['double','Double','Double peaked']:
        def func(r,z):
            f =  gaussian(r=r,z=z,n0=1,a=15,b=0.65,rmax=0.65) + 3*gaussian(r=r,z=z,n0=1,a=35,b=2,rmax=0.45) 
            f*sep_factor(r,z)
            f[z>0.48] = 0
            
            return f
        return func
    
    else:
        raise ValueError(f"Unknown phantom function name: {name}.")