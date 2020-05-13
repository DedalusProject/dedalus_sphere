import numpy             as np
import jacobi            as Jacobi
import scipy.sparse      as sparse

dtype = 'float128'

def quadrature(Lmax,**kwargs):
    """Generates the Gauss quadrature grid and weights for spherical harmonics transform.
        Returns cos_theta, weights
        
        Will integrate polynomials on (-1,+1) exactly up to degree = 2*Lmax+1.
    
    Parameters
    ----------
    Lmax: int >=0; spherical-harmonic degree.
    
    """
    
    return Jacobi.quadrature(Lmax+1,0,0,**kwargs)


def harmonics(Lmax,m,s,cos_theta,dtype=dtype):
    """
        Gives spin-wieghted spherical harmonic functions on the Gauss quaduature grid.
        Returns an array with
            shape = ( Lmax - Lmin(m,s) + 1, len(z) )
                 or (Lmax - Lmin(m,s) + 1,) if z is a single point.
        
        Parameters
        ----------
        Lmax: int >=0; spherical-harmonic degree.
        m,s : int
            spherical harmonic parameters.
        cos_theta: np.ndarray or float.
        dtype: output dtype. internal dtype = 'float128'.
        """
    
    a, b, n = abs(m+s), abs(m-s), Lmax - max(abs(m),abs(s))
    
    init = ((-1.)**max(m,-s))*Jacobi.measure(a,b,cos_theta)**(1/2)
    
    Ylm = Jacobi.polynomials(n+1,a,b,cos_theta,init,dtype='float128')
    
    return Ylm.astype(dtype)

