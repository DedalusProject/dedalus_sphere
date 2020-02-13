import numpy             as np
from . import jacobi128  as jacobi
import scipy.sparse      as sparse


def quadrature(Lmax,**kw):
    """Generates the Gauss quadrature grid and weights for spherical harmonics transform.
        Returns cos_theta, weights
        
        Will integrate polynomials on (-1,+1) exactly up to degree = 2*Lmax+1."""
    
    return jacobi.quadrature(Lmax,0,0,**kw)


def Y(Lmax,m,s,cos_theta):
    """
        Gives spin-wieghted spherical harmonic functions on the Gauss quaduature grid.
        Returns an array with shape = ( Lmax - Lmin(m,s) + 1, len(z) ).
        
        Parameters
        ----------
        m,s : int
        spherical harmonic parameters.
        dm, ds: int
        increments in (m,s)
        a,b : int
        starting Jacobi parameters
        da,db: int
        increments in (a,b)
        
        """
    
    a, b, N = _spin2Jacobi(Lmax,m,s)
    
    init    = ((-1.)**max(m,-s))*jacobi.envelope(a,b,0,0,cos_theta)
    
    return jacobi.recursion(N,a,b,cos_theta,init)


def k_element(mu,ell,s,radius=1):
    return -mu*np.sqrt((ell-mu*s)*(ell+mu*s+1)/2)/radius


def operator(op,Lmax,m,s,radius=1):
    """
        Various derivative and multiplication operators for spin-weighted spherical harmonics .
        
        Parameters
        ----------
        op: string = 'k+', 'k-', 'S+', 'S-', 'I', 'C'
        m, s: int spherical harmonic parameters
        I  = Identity
        k+ = sqrt(0.5)*(Grad_theta + i Grad_phi) with (m,s) -> (m,s+1); diagonal in ell
        k- = sqrt(0.5)*(Grad_theta - i Grad_phi) with (m,s) -> (m,s-1); diagonal in ell
        C  = Cosine multiplication with (m,s) -> (m,s);   couples ell
        S+ = Sine multiplication   with (m,s) -> (m,s+1); couples ell
        S- = Sine multiplication   with (m,s) -> (m,s-1); couples ell
        
        """

    if op in ['D-','D+','S-','S+']:
    
        ds             = int(op[1]+'1')
        a, b, da,db, N = _spin2Jacobi(Lmax,m,s,ds=ds)
        rescale        = -ds*np.sqrt(0.5)/radius
    
        # derivatives
        if (op == 'D+') or (op=='D-'):
            if (da== 1) and (db==-1): return jacobi.operator('C+',N  ,a,b,rescale=rescale)
            if (da==-1) and (db== 1): return jacobi.operator('C-',N  ,a,b,rescale=rescale)
            if (da== 1) and (db== 1): return jacobi.operator('D+',N  ,a,b,rescale=rescale)[:-1,:]
            if (da==-1) and (db==-1): return jacobi.operator('D-',N+1,a,b,rescale=rescale)[:,:-1]
        
        # sine multiplication
        if (op == 'S+') or (op=='S-'):
            
            if (da== 1) and (db==-1):
                A = jacobi.operator('A+',N+1,a,  b)
                B = jacobi.operator('B-',N+1,a+1,b,rescale=ds)
                return (B @ A)[:-1,:-1]
            
            if (da==-1) and (db== 1):
                A = jacobi.operator('A-',N+1,a,  b)
                B = jacobi.operator('B+',N+1,a-1,b,rescale=-ds)
                return (B @ A)[:-1,:-1]
            
            if (da== 1) and (db== 1):
                A = jacobi.operator('A+',N+1,a,  b)
                B = jacobi.operator('B+',N+1,a+1,b,rescale=ds)
                return (B @ A)[:-2,:-1]
            
            if (da==-1) and (db==-1):
                A = jacobi.operator('A-',N+2,a,  b)
                B = jacobi.operator('B-',N+2,a-1,b,rescale=-ds)
                return (B @ A)[:-1,:-2]


    a, b, N = _spin2Jacobi(Lmax,m,s)

    # identity
    if op == 'I': return jacobi.operator('I',N,a,b)
    
    # cosine multiplication
    if op == 'C': return jacobi.operator('J',N,a,b)


def _spin2Jacobi(Lmax,m,s,dm=None,ds=None):
    """
        Converts spherical harmonic parameters (m,s,Lmax) into Jacobi parameters (a,b,n).
        
        Parameters
        ----------
        m,s : int
        spin-weighted spherical harmonic parameters.
        dm, ds: int
        increments in (m,s)
        a,b : int
        Jacobi parameters
        da,db: int
        increments in (a,b) given (dm,ds)
        
        """
    
    a, b, n = abs(m+s), abs(m-s), size(Lmax,m,s)
    
    if (dm == None) and (ds == None): return a,b,n
    
    if dm == None: dm = 0
    if ds == None: ds = 0
    
    da, db, dn = _spin2Jacobi(Lmax,m+dm,s+ds)
    da, db     = da-a, db-b
    
    return a,b,da,db,n

def size(Lmax,m,s): return Lmax - np.max([np.abs(m),np.abs(s)]) + add

def zeros(Lmax,m,s_in,s_out):
    """ non-square array of zeros."""
    Nout, Nin = size(Mmax,m,s_out), size(Lmax,m,s_in)
    return sparse.csr_matrix((Nout+1,Nin+1))


