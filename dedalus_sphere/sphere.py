import numpy             as np
import jacobi            as Jacobi
from scipy.sparse import dia_matrix as banded
from operators    import Operator, infinite_csr

dtype = 'float128'

def quadrature(Lmax,dtype=dtype):
    """Generates the Gauss quadrature grid and weights for spherical harmonics transform.
        Returns cos_theta, weights
        
        Will integrate polynomials on (-1,+1) exactly up to degree = 2*Lmax+1.
    
    Parameters
    ----------
    Lmax: int >=0; spherical-harmonic degree.
    
    """
    
    return Jacobi.quadrature(Lmax+1,0,0,dtype=dtype)


def spin2Jacobi(Lmax,m,s,ds=None,dm=None):

    n    = Lmax + 1 - max(abs(m),abs(s))
    a, b = abs(m+s), abs(m-s)
  
    if ds == dm == None:
        return n,a,b

    if ds == None: ds = 0
    if dm == None: dm = 0

    m += dm
    s += ds
    
    dn    = Lmax + 1 - max(abs(m),abs(s)) - n
    da,db = abs(m+s) - a, abs(m-s) - b

    return n,a,b,dn,da,db


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
    
    n,a,b = spin2Jacobi(Lmax,m,s)
    
    init = np.exp(0.5*Jacobi.measure(a,b,cos_theta,log=True).astype('float128'))
    init *= ((-1.)**max(m,-s))
    
    return Jacobi.polynomials(n,a,b,cos_theta,init,dtype='float128').astype(dtype)


def operator(name,**kwargs):
    """
    Interface to base ShereOperator class.
    
    Parameters
    ----------
    
    """
    
    if name == 'Id':
        return SphereOperator.identity
        
    if name == 'Pi':
        return SphereOperator.parity
    
    if name == 'L':
        return SphereOperator.L()
    
    if name == 'M':
        return SphereOperator.M()
    
    if name == 'S':
        return SphereOperator.S()
    
    if name == 'Cos':
        def Cos(Lmax,m,s):
            return Jacobi.operator('Z')(*spin2Jacobi(Lmax,m,s))
        return Operator(Cos,SphereCodomain(1,0,0,0))
        
    return SphereOperator(name,**kwargs)


class SphereOperator():
    
    def __init__(self,name,radius=1):
            
        self.__function   = getattr(self,f'_SphereOperator__{name}')
        
        self.__radius = radius
            
    def __call__(self,ds):
        return Operator(*self.__function(ds))
    
    @property
    def radius(self):
        return self.__radius
    
    def __D(self,ds):
        
        def D(Lmax,m,s):
            
            n,a,b,dn,da,db = spin2Jacobi(Lmax,m,s,ds=ds)
            
            D = Jacobi.operator('C' if da+db == 0 else 'D')(da)
            
            return  (-ds*np.sqrt(0.5)/self.radius)*D(n,a,b)
    
        return D, SphereCodomain(0,0,ds,0)
    
    def __Sin(self,ds):

        def Sin(Lmax,m,s):

            n,a,b,dn,da,db = spin2Jacobi(Lmax,m,s,ds=ds)
            
            print(da,db,n,a,b)
            
            S = Jacobi.operator('A')(da) @ Jacobi.operator('B')(db)

            return (da*ds) * S(n,a,b)

        return Sin, SphereCodomain(1,0,ds,0)
    
    @staticmethod
    def identity(dtype=dtype):
        
        def I(Lmax,m,s):
            n = spin2Jacobi(Lmax,m,s)[0]
            N = np.ones(n,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))
            
        return Operator(I,SphereCodomain(0,0,0,0))
        
    @staticmethod
    def parity(dtype=dtype):
        
        def Pi(Lmax,m,s):
            return Jacobi.operator('Pi')(*spin2Jacobi(Lmax,m,s))
            
        return Operator(Pi,SphereCodomain(0,0,0,1))
    
    @staticmethod
    def L(dtype=dtype):
        
        def L(Lmax,m,s):
            n = spin2Jacobi(Lmax,m,s)[0]
            N = np.arange(Lmax+1-n,Lmax+1,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))
            
        return Operator(L,SphereCodomain(0,0,0,0))
    
    @staticmethod
    def M(dtype=dtype):
        
        def M(Lmax,m,s):
            n = spin2Jacobi(Lmax,m,s)[0]
            N = m*np.ones(n,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))
            
        return Operator(M,SphereCodomain(0,0,0,0))
    
    @staticmethod
    def S(dtype=dtype):
        
        def S(Lmax,m,s):
            n = spin2Jacobi(Lmax,m,s)[0]
            N = s*np.ones(n,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))
            
        return Operator(S,SphereCodomain(0,0,0,0))
    

class SphereCodomain():
    
    def __init__(self,dL=0,dm=0,ds=0,pi=0):
        self.__arrow = (dL,dm,ds,pi)
        
    @property
    def arrow(self):
        return self.__arrow
    
    def __getitem__(self,item):
        return self.__arrow[(item)]
    
    def __str__(self):
        s = f'(L->L+{self[0]},m->m+{self[1]},s->s+{self[2]})'
        if self[3]: s = s.replace('s->s','s->-s')
        return s.replace('+0','').replace('+-','-')
        
    def __repr__(self):
        return str(self)
    
    def __add__(self,other):
        return SphereCodomain(*self(*other[:3],evaluate=False),self[3]^other[3])
    
    def __call__(self,*args,evaluate=True):
        L,m,s = args[:3]
        if self[3]: s *= -1
        return self[0] + L, self[1] + m, self[2] + s
    
    def __eq__(self,other):
        return self[1:] == other[1:]
    
    def __or__(self,other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        if self[0] >= other[0]:
            return self
        return other
    
    def __neg__(self):
        m,s = -self[1],-self[2]
        if self[3]: s *= -1
        return SphereCodomain(-self[0],m,s,self[3])
    
    def __mul__(self,other):
        if type(other) != int:
            raise TypeError('only integer multiplication defined.')
        
        if other == 0:
            return SphereCodomain()
        
        if other < 0:
            return -self + (other+1)*self
            
        return self + (other-1)*self
    
    def __rmul__(self,other):
        return self*other
    
    def __sub__(self,other):
        return self + (-other)
    

