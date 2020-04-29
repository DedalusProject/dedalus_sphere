import numpy             as np
import scipy.sparse      as sparse

from operators import *

dtype='float128'

default = lambda f: f
dense   = lambda f: f.todense()
banded  = sparse.dia_matrix
row     = sparse.csr_matrix
column  = sparse.csc_matrix

formatter = row

def format(func):
    def wrapper(*args):
        return formatter(func(int(args[0]+1),*args[1:]))
    wrapper.__name__ = func.__name__
    return wrapper


def polynomials(n,a,b,z,init=None,normalised=True,dtype=dtype,Newton=False):

    if init == None:
        init = 1 + 0*z
        if normalised:
            init /= np.sqrt(mass(a,b),dtype='float128')
    
    Z = operator('Z',normalised=normalised,dtype='float128')
    Z = banded(Z(n,a,b).T).data
    
    if type(z) == np.ndarray:
        shape = (n+1,len(z))
    else:
        shape = n+1
    
    P     = np.zeros(shape,dtype='float128')
    P[0]  = init

    if len(Z) == 2:
        P[1]  = z*P[0]/Z[1,1]
        for k in range(2,n+1):
            P[k] = (z*P[k-1] - Z[0,k-2]*P[k-2])/Z[1,k]
    else:
        P[1]  = (z-Z[1,0])*P[0]/Z[2,1]
        for k in range(2,n+1):
            P[k] = ((z-Z[1,k-1])*P[k-1] - Z[0,k-2]*P[k-2])/Z[2,k]
    
    if Newton:
        L = n + (a+b)/2
        return z + (1-z**2)*P[n-1]/(L*Z[-1,n]*P[n]-(L-1)*Z[0,n-2]*P[n-2]), P[:n-1].astype(dtype)

    return P[:n].astype(dtype)


def quadrature(n,a,b,days=3,normalised=True,dtype=dtype):
    
    z = grid_guess(n,a,b,dtype=dtype)
    
    for _ in range(days):
        z, P = polynomials(n+1,a,b,z,Newton=True,normalised=normalised)
    
    P[0] /= np.sqrt(np.sum(P**2,axis=0))
    w = P[0]**2
    
    if not normalised:
        w *= mass(a,b)
    
    return z.astype(dtype), w.astype(dtype)


def grid_guess(n,a,b,dtype=dtype):
    return np.cos(np.pi*(np.arange(4*n-1,2,-4,dtype=dtype)+2*a)/(4*n+2*(a+b+1)))
 
 
def operator(name,normalised=True,dtype=dtype):
    if name == 'Id':
        return JacobiOperator.identity(dtype=dtype)
    if name == 'Pi':
        return JacobiOperator.parity(dtype=dtype)
    if name == '0':
        return JacobiOperator.zero(dtype=dtype)
    if name == 'Z':
        A = JacobiOperator('A',normalised=normalised,dtype=dtype)
        B = JacobiOperator('B',normalised=normalised,dtype=dtype)
        return (B(-1) @ B(+1) - A(-1) @ A(+1))/2
    return JacobiOperator(name,normalised=normalised,dtype=dtype)
   
   
def measure(z,a,b,normalised=True,log=False):

    if not log:
        w = (1-z)**a * (1+z)**b
        if normalised: w /= mass(a,b)
        return w
        
    if a <= 1 and b <= 1:
        return np.log(measure(z,a,b,normalised=normalised))

    ia, ib = int(a), int(b)

    a, b = a - ia, b - ib

    S = ia*np.log(1-z) + ib*np.log(1+z) + measure(z,a,b,normalised=False,log=True)
    
    if normalised: S -= mass(a+ia,b+ib,log=True)
    return S


def mass(a,b,log=False):

    if not log:
        from scipy.special import beta
        return 2**(a+b+1)*beta(a+1,b+1)

    from scipy.special import betaln
    return (a+b+1)*np.log(2) + betaln(a+1,b+1)


def norm_ratio(dn,da,db,n,a,b,squared=False):

    if not all(type(d) == int for d in (dn,da,db)):
        raise TypeError('can only increment by integers.')
    
    def n_ratio(d,n,a,b):
        if d <  0: return 1/n_ratio(-d,n+d,a,b)
        if d == 0: return 1 + 0*n
        if d == 1:
            if a+b == -1: return (n+a+1)*(n+b+1)/(n+1)**2
            return (n+a+1)*(n+b+1)*(2*n+a+b+1)/((n+1)*(n+a+b+1)*(2*n+a+b+3))
        return n_ratio(1,n+d-1,a,b)*n_ratio(d-1,n,a,b)
    
    def ab_ratio(d,n,a,b):
        if d <  0: return 1/ab_ratio(-d,n,a+d,b)
        if d == 0: return 1 + 0*n
        if d == 1:
            if a+b == -1: return 4*(n+a+1)/(2*n+1)
            return 2*(n+a+1)*(2*n+a+b+1)/((n+a+b+1)*(2*n+a+b+2))
        return ab_ratio(1,n,a+d-1,b)*ab_ratio(d-1,n,a,b)

    ratio = n_ratio(dn,n,a+da,b+db)*ab_ratio(da,n,a,b+db)*ab_ratio(db,n,b,a)
    
    if not squared:
        return np.sqrt(ratio)
    return ratio
        
    
class JacobiOperator():
    
    dtype='float128'
    
    def __init__(self,name,normalised=True,dtype=dtype):
        
        self.__name   = name
        self.__func   = getattr(self,f'_{self.__class__.__name__}__{name}')
        self.__normed = normalised
        
        
    def __call__(self,p):
        return Operator(*self.__func(p))
    
    @property
    def name(self):
        return self.__name
    
    @property
    def normalised(self):
        return self.__normed
    
    def __A(self,p):
        
        @format
        def A(n,a,b):
            
            N = np.arange(n,dtype=self.dtype)
            bands = np.array({+1:[N+(a+b+1),  -(N+b)],
                              -1:[2*(N+a)  ,-2*(N+1)]}[p])
                        
            bands[:,0] = 1/2 if a+b == -1 else bands[:,0]/(a+b+1)
            bands[:,1:] /= 2*N[1:]+a+b+1
            
            if self.normalised:
                bands[0] *= norm_ratio(0,p,0,N,a,b)
                bands[1,(1+p)//2:] *= norm_ratio(-p,p,0,N[(1+p)//2:],a,b)
        
            return banded((bands,[0,p]),(n+(1-p)//2,n))
        
        return A, np.array([(1-p)//2,p,0])

    def __B(self,p):
        
        @format
        def B(n,a,b):
            
            N = np.arange(n,dtype=self.dtype)
            bands = np.array({+1:[N+(a+b+1),   N+a],
                              -1:[2*(N+b)  ,2*(N+1)]}[p])
            
            bands[:,0] = 1/2 if a+b == -1 else bands[:,0]/(a+b+1)
            bands[:,1:] /= 2*N[1:]+a+b+1
            
            if self.normalised:
                bands[0] *= norm_ratio(0,0,p,N,a,b)
                bands[1,(1+p)//2:] *= norm_ratio(-p,0,p,N[(1+p)//2:],a,b)
                
            return banded((bands,[0,p]),(n+(1-p)//2,n))

        return B, np.array([(1-p)//2,0,p])
        
    def __C(self,p):
        
        @format
        def C(n,a,b):
            
            N = np.arange(n,dtype=self.dtype)
            bands = np.array([N + {+1:b,-1:a}[p]])
            
            if self.normalised:
                bands[0] *= norm_ratio(0,p,-p,N,a,b)
            
            return banded((bands,[0]),(n,n))
        
        return C, np.array([0,p,-p])

    def __D(self,p):
        
        @format
        def D(n,a,b):
            
            N = np.arange(n,dtype=self.dtype)
            bands = np.array([(N + {+1:a+b+1,-1:1}[p])*2**(-p)])
            
            if self.normalised:
                bands[0,(1+p)//2:] *= norm_ratio(-p,p,p,N[(1+p)//2:],a,b)
            
            return banded((bands,[p]),(n-p,n))
        
        return D, np.array([-p,p,p])
        
    
    @staticmethod
    def identity(dtype=dtype):
        
        @format
        def I(n,a,b):
        
            N = np.ones(n,dtype=dtype)
            return banded((N,[0]),(n,n))
            
        return Operator(I,np.array([0,0,0]))
    
    
    @staticmethod
    def parity(dtype=dtype):
        
        @format
        def P(n,a,b):
        
            N = np.arange(n,dtype=dtype)
            return banded(((-1)**N,[0]),(n,n))
        
        # a,b -> b,a , z -> -z
        # The arrow is not additive.
        # which is not implimented in Operator.
        # In general: parity @ A(p) @ parity ==  B(p)
        #             parity @ C(p) @ parity ==  C(-p)
        #             parity @ D(p) @ parity == -p*D(p)
        return Operator(P, np.array([0,0,0]))
        
        
    @staticmethod
    def zero(dtype=dtype):
        
        @format
        def Z(n,a,b):
        
            N = np.zeros(n,dtype=JacobiOperator.dtype)
            return banded((N,[0]),(n,n))
            
        return Operator(Z,np.array([0,0,0]))
