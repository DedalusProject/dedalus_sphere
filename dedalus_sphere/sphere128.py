import numpy             as np
from . import jacobi128         as jacobi
import scipy.sparse      as sparse

def a_and_b(m,s,dm=None,ds=None):
    """
    Converts spherical harmonic parameters (m,s) into Jacobi parameters (a,b).

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

    a,b = abs(m+s),abs(m-s)

    if (dm == None) and (ds == None): return a,b

    if (dm == None): dm = 0
    if (ds == None): ds = 0

    da,db = a_and_b(m+dm,s+ds)
    da,db = da-a, db-b

    return a,b,da,db

def L_min(m,s):
    """ Minimum spherical harmonic degree depends on (m,s). """
    return max(abs(m),abs(s))

def quadrature(L_max,**kw):
    """ Generates the Gauss quadrature grid and weights for spherical harmonics transform.
        Returns cos_theta, weights

        Parameters
        ----------
        L_max: int
        Will integrate polynomials on (-1,+1) exactly up to degree = 2*L_max+1.

    """

    return jacobi.quadrature(L_max,0,0,**kw)

def Y(L_max,m,s,cos_theta):
    """
        Gives spin-wieghted spherical harmonic functions on the Gauss quaduature grid.
        Returns an array with shape = ( L_max-L_min(m,s) + 1, len(cos_theta) ).

        Parameters
        ----------
        L_max, m,s : int
        spherical harmonic parameters.
        dm, ds: int
        increments in (m,s)
        a,b : int
        starting Jacobi parameters
        da,db: int
        increments in (a,b)

    """
    a, b  = a_and_b(m,s)
    N     = L_max - L_min(m,s)
    phase = (-1)**int(max(m,-s))

    init  = phase*jacobi.envelope(a,b,0,0,cos_theta)

    return jacobi.recursion(N,a,b,cos_theta,init)

def unitary(rank=1,adjoint=False):
    """ Transforms the components of vectors and tensors.
        U:        (v[th],v[ph]) --> (v[+],v[-])
        Uadjoint: (v[+],v[-])   --> (v[th],v[ph])

        Parameters
        ----------
        rank: int
        rank=1 for vectors, rank=2 for matrices, etc
        adjoint: T/F
        returns the inverse transformation

    """

    if rank == 0: return 1

    if adjoint :
        U       = np.sqrt(0.5)*np.array([[1,1j],[1,-1j]])
    else:
        U       = np.sqrt(0.5)*np.array([[1,1],[-1j,1j]])
    unitary = U
    for k in range(rank-1):
        unitary = np.kron(U,unitary)

    return unitary

def operator(op,L_max,m,s):
    """ Various derivative and multiplication operators for spin-weighted spherical harmonics .

        Parameters
        ----------
        L_max, m, s: int
        spherical harmonic parameters
        op: string = 'I', 'C', 'k+', 'k-', 'S+', 'S-'
        I  = Identity
        k+ = sqrt(0.5)*(Grad_theta + i Grad_phi) with (m,s) -> (m,s+1); diagonal in ell
        k- = sqrt(0.5)*(Grad_theta - i Grad_phi) with (m,s) -> (m,s-1); diagonal in ell
        C  = Cosine multiplication with (m,s) -> (m,s);   couples ell
        S+ = Sine multiplication   with (m,s) -> (m,s+1); couples ell
        S- = Sine multiplication   with (m,s) -> (m,s-1); couples ell

    """

    def ds(op):
        """get s increment from the operator string"""
        if len(op)==1: return 0
        return int(op[1]+'1')

    mu, N     = ds(op), L_max-L_min(m,s)
    a,b,da,db = a_and_b(m,s,ds=mu)
    rescale   = -mu*np.sqrt(0.5)

    # identity
    if op == 'I':
        a,b  = a_and_b(m,s)
        return jacobi.operator('I',N,a,b)

    # cosine multiplication
    if op == 'C':
        a,b  = a_and_b(m,s)
        return jacobi.operator('J',N,a,b)

    # derivatives
    if (op == 'k+') or (op=='k-'):
        if (da== 1) and (db==-1): return jacobi.operator('C+',N  ,a,b,rescale=rescale)
        if (da==-1) and (db== 1): return jacobi.operator('C-',N  ,a,b,rescale=rescale)
        if (da== 1) and (db== 1): return jacobi.operator('D+',N  ,a,b,rescale=rescale)[:-1,:]
        if (da==-1) and (db==-1): return jacobi.operator('D-',N+1,a,b,rescale=rescale)[:,:-1]

    # sine multiplication
    if (op == 'S+') or (op=='S-'):
        if (da== 1) and (db==-1):
            A = jacobi.operator('A+',N+1,a,  b)
            B = jacobi.operator('B-',N+1,a+1,b)
            return mu*(B.dot(A))[:-1,:-1]
        if (da==-1) and (db== 1):
            A = jacobi.operator('A-',N+1,a,  b)
            B = jacobi.operator('B+',N+1,a-1,b)
            return -mu*(B.dot(A))[:-1,:-1]
        if (da== 1) and (db== 1):
            A = jacobi.operator('A+',N+1,a,  b)
            B = jacobi.operator('B+',N+1,a+1,b)
            return mu*(B.dot(A))[:-2,:-1]
        if (da==-1) and (db==-1):
            A = jacobi.operator('A-',N+2,a,  b)
            B = jacobi.operator('B-',N+2,a-1,b)
            return -mu*(B.dot(A))[:-1,:-2]

def zeros(L_max,m,s_out,s_in):
    """ non-square array of zeros.""" # Cannot make an operator because of non-square.
    return sparse.csr_matrix((L_max+1-L_min(m,s_out),L_max+1-L_min(m,s_in)))




