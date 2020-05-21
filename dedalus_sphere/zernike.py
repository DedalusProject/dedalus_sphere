import numpy as np
import jacobi  as Jacobi
from  jacobi import JacobiCodomain
from operators    import Operator, infinite_csr


# The defalut configuration for the base Jacobi parameter.
alpha = 0

def quadrature(dimension,n,k=alpha):
    """
    Weights associated with
        dV = (1-r*r)**k * r**(dimension-1) dr, where 0 <= r <= 1.
        
    """
    
    z, w = Jacobi.quadrature(n,k,dimension/2 - 1)
    
    w /= 2**( k + dimension/2 + 1 )
    
    return z, w

def regularity2Jacobi(dimension,n,k,ell):
    
    n = n - nmin(ell)
    a = k
    b = ell + dimension/2 - 1
    
    return n,a,b
    
def nmin(ell):
    return max(ell//2,0)


def polynomials(dimension,n,k,ell,z):
    """
        Unit normalised:
            
            integral(Q**2 dV)  = 1
    
    """
    
    init  = Jacobi.measure(0,ell,z,log=True,probability=False)
    
    ell += dimension/2 - 1
    
    init -= Jacobi.mass(k,ell,log=True)  - np.log(2)*(k + dimension/2 + 1)
    init = np.exp(0.5*init)
    
    return Jacobi.polynomials(n,k,ell,z,init)


def operator(dimension,name):
    """
    Interface to base ZernikeOperator class.

    Parameters
    ----------

    """
    
    if name == 'Z':
        def Z(n,k,ell):
            ell += dimension/2 - 1
            return Jacobi.operator('Z')(n,k,ell)
        return Operator(Z,ZernikeCodomain(1,0,0))
        
    return ZernikeOperator(dimension,name)

class ZernikeOperator():
    
    def __init__(self,dimension,name,radius=1):
            
        self.__function   = getattr(self,f'_ZernikeOperator__{name}')
        
        self.__dimension = dimension
        self.__radius    = radius
        
            
    def __call__(self,p):
        return Operator(*self.__function(p))
    
    @property
    def dimension(self):
        return self.__dimension
    
    @property
    def radius(self):
        return self.__radius
    
    def __D(self,dl):
        
        def D(n,k,ell):
            
            D = Jacobi.operator('D' if dl > 0 else 'C')(+1)
            
            ell += self.dimension/2 - 1
            
            return  (2/self.radius)*D(n,k,ell)
    
        return D, ZernikeCodomain(-(1+dl)//2,1,dl)
        
        
    def __E(self,dk):
        
        def E(n,k,ell):
            
            ell += self.dimension/2 - 1
            
            return  np.sqrt(0.5)*Jacobi.operator('A')(dk)(n,k,ell)
    
        return E, ZernikeCodomain((1-dk)//2,dk,0)
        
    
    def __R(self,dl):

        def R(n,k,ell):

            ell += self.dimension/2 - 1
            
            return (np.sqrt(0.5)*self.radius)*Jacobi.operator('B')(dl)(n,k,ell)

        return R, ZernikeCodomain((1-dl)//2,0,dl)
    
    
class ZernikeCodomain(JacobiCodomain):

    def __init__(self,dn=0,dk=0,dl=0,pi=0):
        JacobiCodomain.__init__(self,dn,dk,dl,0,Output=ZernikeCodomain)
    
    def __str__(self):
        s = f'(n->n+{self[0]},k->k+{self[1]},l->l+{self[2]})'
        return s.replace('+0','').replace('+-','-')
        
