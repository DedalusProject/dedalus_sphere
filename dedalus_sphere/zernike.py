import numpy as np
import jacobi  as Jacobi
from  jacobi import JacobiCodomain

# The defalut configurations for the base Jacobi parameter.
alpha = 0

def quadrature(dimension,n,**kwargs):
    """
    Weights associated with
        dV = (1-r*r)**alpha * r**(dimension-1) dr, where 0 <= r <= 1.
        
    """
    
    z, w = Jacobi.quadrature(*regularity2Jacobi(dimension,n,0,0,0,**kwargs))
    
    w /= 2**( alpha + dimension/2 + 1 )
    
    return z, w


def regularity2Jacobi(dimension,n,k,ell,degree,alpha=alpha):
    
    if type(degree) != tuple: degree = (degree,degree)
    
    a = k + alpha
    b = ell + degree[0] + dimension/2 - 1
    n = n - nmin(ell,degree[1])

    return n,a,b

def nmin(ell,degree):
    return max((ell + degree)//2,0)

def polynomials(dimension,n,ell,degree,z,alpha=alpha):
    """
        Unit normalised:
            
            integral(Q**2 dV)  = 1
    
    """
    
    a, b = regularity2Jacobi(dimension,n,0,ell,degree,alpha=alpha)[1:]

    init  = Jacobi.measure(0,ell+degree,z,log=True)
    init -= Jacobi.mass(a,b,log=True)  - np.log(2)*(dimension/2 + 2)
    
    return Jacobi.polynomials(n,a,b,z,np.exp(0.5*init))


def operator(dimension,name,**kwargs):
     
    #    Ball : (Jacobi,rescale)
    O = {'D-' : ('C+',np.sqrt(8.0)/radius),   # d/dr + (ell+1)/r
         'D+' : ('D+',np.sqrt(8.0)/radius),   # d/dr -  ell/r
         'R-' : ('B-',np.sqrt(0.5)*radius),   # r multiplication
         'R+' : ('B+',np.sqrt(0.5)*radius),   # r multiplication
          'Z' : ('J',1),                      # z = 2*(r/R)**2 - 1 multiplication
          'I' : ('I',1),                      # identity
          'E' : ('A+',1),                     # conversion
          '0' : ('0',1),                      # zeros
        'r=R' : ('z=+1',2**((ell+degree)/2))} # boundary restriction
                                      
    a, b, N = _regularity2Jacobi(dimension,Nmax+pad,k,ell,(degree,0),alpha=alpha)

    return Jacobi.operator(O[op][0],N,a,b,rescale=O[op][1])


class ZernikeCodomain(JacobiCodomain):

    def __init__(self,dn=0,dk=0,dl=0):
        JacobiCodomain.__init__(self,dn,dk,dl,0,output=ZernikeCodomain)
    
    def __str__(self):
        s = f'(n->n+{self[0]},k->k+{self[1]},l->l+{self[2]})'
        return s.replace('+0','').replace('+-','-')
        
