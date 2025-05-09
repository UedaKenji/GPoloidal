import os,sys
import numpy as np
from typing import Callable, Optional, Tuple, Union, List
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import mpl_toolkits.axes_grid1

import numpy.typing as npt

import scipy.sparse as sparse

from . import plot_utils
import zray


class ImageVector(np.ndarray):
    def __new__(cls, input_array, im_shape,mask=None):
        obj = np.asarray(input_array).view(cls)
        if mask is not None:
            if np.prod(im_shape)  != obj.size + np.sum(mask):
                raise ValueError("im_shape does not match the size of input array")
        else:
            if np.prod(im_shape) != obj.size:
                raise ValueError("im_shape does not match the size of input array")
        obj.im_shape = im_shape  # 画像の形状を保存
        obj.mask = mask
        return obj

    @property
    def im(self):
        """2次元にリシェイプされた画像を取得"""
        if self.mask is None:
            return self.reshape(self.im_shape)
        else:
            im = np.empty(self.size + np.sum(self.mask))
            im[~self.mask] = self
            im[self.mask] = np.nan
            return im.reshape(self.im_shape)

    def flatten(self):
        """1次元に変換"""
        return super().flatten()

class sparse_matrix:
    """
    スパース行列のクラス
    """
    def __init__(self, 
        H: np.ndarray|sparse.csr_matrix,
        ray:zray.Ray,
        im_shape: Tuple[int, int] = None,
        mask: Optional[npt.NDArray[np.bool_]] = None,
        ):

        self.ray = ray
        if type(H) == np.ndarray:
            self.Mat = sparse.csr_matrix(H)
        elif type(H) == sparse.csr_matrix:
            self.Mat = H
        else:
            raise ValueError("H must be numpy.ndarray or sparse.csr_matrix")
        
        if im_shape is not None:
            if np.prod(im_shape) != self.Mat.shape[0]:
                raise ValueError("im_shape does not match the size of input array")
        else:
            im_shape = None
        
        self.mask = mask
        self.im_shape = im_shape
        print( f"density : {self.Mat.size/(self.ray.M*self.Mat.shape[1]):.4f}")

    def __matmul__(self, other):
        """
        スパース行列とベクトルの積を計算する
        """
        res =  self.Mat @ other
        if self.im_shape is not None and len(other.shape) == 1:
            res = ImageVector(res, self.im_shape, mask=self.mask)        
        return res
    
    def set_mask(self, mask: npt.NDArray[np.bool_]):
        """
        マスクを設定する
        """
        self.mask = mask
        self.Mat0 = self.Mat.copy()
        self.Mat = sparse.csr_matrix(self.Mat.toarray()[~mask])

    def unset_mask(self):
        """
        マスクを解除する
        """
        self.mask = None
        self.Mat = self.Mat0.copy()
        self.Mat0 = None


typical_ref_index_param_in_rt1 = {
    '400nm':{'n_R':1.7689 ,'n_I':0.60521},
    '700nm':{'n_R':0.70249,'n_I':0.36890}
    }

def refractive_indices_for_metal(
    cos_i: float|np.ndarray, 
    n_R  : float|np.ndarray = 1.7689, #for 400nm
    n_I  : float|np.ndarray = 0.60521 #for 400nm
    ) -> float|np.ndarray:
    """""
    金属の反射率を計算する．
    :param cos_i: cos θ 
    :param n_R: 屈折率の実数部
    :param n_I: 屈折率の虚数部（消光係数）
    :return: s偏光の反射率（絶対値），p偏光の反射率（絶対値）
    """""
    sin_i = np.sqrt(1-cos_i**2)

    r_TE = (cos_i - np.sqrt((n_R**2 - n_I**2 - sin_i**2) + 2j*n_I*n_R))\
          /(cos_i + np.sqrt((n_R**2 - n_I**2 - sin_i**2) + 2j*n_I*n_R))
    r_TM = (-(n_R**2 - n_I**2 + 2j*n_R*n_I)*cos_i + np.sqrt((n_R**2 - n_I**2 - sin_i**2) + 2j*n_I*n_R))\
          /((n_R**2 - n_I**2 + 2j*n_R*n_I) *cos_i + np.sqrt((n_R**2 - n_I**2 - sin_i**2) + 2j*n_I*n_R))
    return (np.abs(r_TE)**2+ np.abs(r_TM)**2)/2 # type: ignore


class Integrated_matrix:
    """
    複数のスパース行列をまとめて計算するクラス
    """
    def __init__(self, 
        H_list: List[sparse_matrix]
        ):
        self.H_list = H_list
        self.nH = len(H_list)

    def set_mask(self, mask: npt.NDArray[np.bool_]):
        """
        マスクを設定する
        """
        for H in self.H_list:
            H.set_mask(mask)    
    
    def unset_mask(self):
        """
        マスクを解除する
        """
        for H in self.H_list:
            H.unset_mask()

    def set_reflection(self, reflection: npt.NDArray[np.bool_]):
        """
        反射を設定する
        """
        for H in self.H_list:
            H.set_reflection(reflection)    

