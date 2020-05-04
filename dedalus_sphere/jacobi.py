import numpy             as np
from scipy.sparse import dia_matrix as banded

from operators import infinite_csr, Operator

dtype='float128'

def format(func):
    def wrapper(*args):
        return infinite_csr(func(*args))
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
        
        self.__name       = name
        self.__normalised = normalised
        self.__function   = getattr(self,f'_JacobiOperator__{name}')
                                                  
    @property
    def name(self):
        return self.__name
    
    @property
    def normalised(self):
        return self.__normalised
    
    def __call__(self,p):
        return Operator(*self.__function(p))
    
    def __A(self,p):
        
        @format
        def A(n,a,b):
            if a+p <= -1:
                return banded((n+(1-p)//2,n))
            N = np.arange(n,dtype=self.dtype)
            bands = np.array({+1:[N+(a+b+1),  -(N+b)],
                              -1:[2*(N+a)  ,-2*(N+1)]}[p])
            bands[:,0] = 1/2 if a+b == -1 else bands[:,0]/(a+b+1)
            bands[:,1:] /= 2*N[1:]+a+b+1
            if self.normalised:
                bands[0] *= norm_ratio(0,p,0,N,a,b)
                bands[1,(1+p)//2:] *= norm_ratio(-p,p,0,N[(1+p)//2:],a,b)
            return banded((bands,[0,p]),(n+(1-p)//2,n))
        
        return A, JacobiCodomain((1-p)//2,p,0,0)

    def __B(self,p):
        
        @format
        def B(n,a,b):
            if b+p <= -1:
                return banded((n+(1-p)//2,n))
            N = np.arange(n,dtype=self.dtype)
            bands = np.array({+1:[N+(a+b+1),   N+a],
                              -1:[2*(N+b)  ,2*(N+1)]}[p])
            bands[:,0] = 1/2 if a+b == -1 else bands[:,0]/(a+b+1)
            bands[:,1:] /= 2*N[1:]+a+b+1
            if self.normalised:
                bands[0] *= norm_ratio(0,0,p,N,a,b)
                bands[1,(1+p)//2:] *= norm_ratio(-p,0,p,N[(1+p)//2:],a,b)
            return banded((bands,[0,p]),(n+(1-p)//2,n))

        return B, JacobiCodomain((1-p)//2,0,p,0)
        
    def __C(self,p):
        
        @format
        def C(n,a,b):
            if a+p <= -1 or b-p <= -1:
                return banded((n,n))
            N = np.arange(n,dtype=self.dtype)
            bands = np.array([N + {+1:b,-1:a}[p]])
            if self.normalised:
                bands[0] *= norm_ratio(0,p,-p,N,a,b)
            return banded((bands,[0]),(n,n))
        
        return C, JacobiCodomain(0,p,-p,0)

    def __D(self,p):
        
        @format
        def D(n,a,b):
            if a+p <= -1 or b+p <= -1:
                return  banded((n-p,n))
            N = np.arange(n,dtype=self.dtype)
            bands = np.array([(N + {+1:a+b+1,-1:1}[p])*2**(-p)])
            if self.normalised:
                bands[0,(1+p)//2:] *= norm_ratio(-p,p,p,N[(1+p)//2:],a,b)
            return banded((bands,[p]),(n-p,n))
        
        return D, JacobiCodomain(-p,p,p,0)
        
    @staticmethod
    def identity(dtype=dtype):
        
        @format
        def I(n,a,b):
            N = np.ones(n,dtype=dtype)
            return banded((N,[0]),(n,n))
            
        return Operator(I,JacobiCodomain(0,0,0,0))
    
    @staticmethod
    def parity(dtype=dtype):
        
        @format
        def P(n,a,b):
            N = np.arange(n,dtype=dtype)
            return banded(((-1)**N,[0]),(n,n))
        
        return Operator(P,JacobiCodomain(0,0,0,1))
        
        
class JacobiCodomain():
    
    def __init__(self,dn,da,db,pi):
        self.__map = (dn,da,db,pi)
    
    def __getitem__(self,item):
        return self.__map[(item)]
    
    def __str__(self):
        return str(self[:])
    
    def __repr__(self):
        return str(self)
    
    def __add__(self,other):
        return JacobiCodomain(*self(*other[0:3]),self[3]^other[3])
        
    def __call__(self,*args):
        n,a,b = args
        if self[3]: a,b = b,a
        return self[0] + n, self[1] + a, self[2] + b
    
    def __eq__(self,other):
        return self[1:] == other[1:]
    
    def __or__(self,other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        if self[0] >= other[0]:
            return self
        return other
    
