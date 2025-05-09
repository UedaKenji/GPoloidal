import numpy as np


    

class Diag:
    """
    Specialized class for diagonal matrices with numpy functionality.
    
    Attributes:
      a (np.ndarray): 1D array holding the diagonal elements.
      
    Constructor:
      Diag(a): Create a diagonal matrix object from a 1D array 'a'.
      
    Methods:
      __matmul__(other): Implements left multiplication (self @ other) for 1D and 2D arrays.
      __rmatmul__(other): Implements right multiplication (other @ self) for 1D and 2D arrays.
      inv(): Returns a new Diag object with each element being the reciprocal of the corresponding diagonal element.
      
    Properties:
      mat: Returns a standard 2D NumPy diagonal matrix.
      
    Also, __array__ is implemented so that np.array(D) returns the 2D matrix representation.
    
    The __getattr__ method is implemented so that any attribute (such as 'mean') not found on Diag is forwarded to the underlying 1D array.
    """
    
    __array_priority__ = 1000  # Increase priority so custom methods are preferred in binary ops.
    
    def __init__(self, a):
        self.d = np.asarray(a)
        if self.d.ndim != 1:
            raise ValueError("Input array must be 1-dimensional.")
            
    def __matmul__(self, other):
        """
        Implements left multiplication: self @ other.
        
        - For a 1D array 'other': perform element-wise multiplication.
        - For a 2D array 'other': scale each row i by self.a[i].
        """
        other = np.asarray(other)
        if other.ndim == 1:
            if other.shape[0] != self.d.shape[0]:
                raise ValueError("Matrix multiplication not defined (dimension mismatch).")
            return self.d * other
        elif other.ndim == 2:
            if other.shape[0] != self.d.shape[0]:
                raise ValueError("Matrix multiplication not defined (dimension mismatch).")
            return self.d[:, np.newaxis] * other
        else:
            raise ValueError("Only 1D or 2D arrays are supported.")
            
            
    def inv(self):
        """
        Returns a new Diag object with each diagonal element being the reciprocal of the original.
        Raises ZeroDivisionError if any diagonal element is zero.
        """
        if np.any(self.d == 0):
            raise ZeroDivisionError("Cannot compute inverse: diagonal element equals zero.")
        return Diag(1.0 / self.d)
    
    @property
    def mat(self):
        """
        Returns the standard 2D diagonal matrix representation.
        """
        return np.diag(self.d)
        
    def __array__(self, dtype=None):
        """
        When conversion with np.array() is requested, return the 2D diagonal matrix.
        """
        if dtype:
            return self.mat.astype(dtype)
        return self.mat
    
    def __getattr__(self, attr):
        """
        Delegate attribute access to the underlying 1D array self.a.
        
        For example, D.mean() will be resolved as D.a.mean().
        """
        return getattr(self.d, attr)
    
    def __repr__(self):
        return f"Diag({self.d})"
    
    

def MAE(test,orignal,Weight=None,normalize=True):
    if Weight is  None: Weight = 1

    if normalize:
        return np.mean(np.abs(test-orignal)*Weight)/np.mean(np.abs(orignal)*Weight)
    else:
        return np.mean(np.abs(test-orignal)*Weight)
    
def RMSE(test,orignal,Weight=None,normalize=True):
    if Weight is  None: Weight = 1

    if normalize:
        return np.sqrt(np.mean((test-orignal)**2*Weight))/np.sqrt(np.mean(np.abs(orignal)**2*Weight))
    else:
        return np.sqrt(np.mean((test-orignal)**2)*Weight)
