from intertwiner import *
from itertools import permutations

# Helper functions
dual   = lambda t: tuple(-e for e in t)
apply  = lambda p: lambda t: tuple(t[i] for i in p)
sum_   = lambda k: lambda t: sum(t[i] for i in k if 0 <= i < len(t))
remove = lambda k: lambda t: tuple(s for i,s in enumerate(t) if not i in k)

def log_(d,n,add=0):
    while n>1:
        n //= d
        add += 1
    return add

def array_check(func):
    def wrapper(self,other):
        if type(other) == np.ndarray:
            n    = other.shape[0]
            rank = log_(self.dimension,n)
            if n != self.dimension**rank:
                raise TypeError('incompatible domain.')
            return eval(f"np.ndarray.{func.__name__}(self(rank),other)")
        return func(self,other)
        wrapper.__name__ = func.__name__
    return wrapper
    
class SpinOperator(object):
    
    def __init__(self,func,arrow,spins=(-1,0,1)):
        
        good = False
        for s in [(-1,1),(-1,0,1)]:
            good = good or spins in list(permutations(s))
        if not good: raise TypeError('invalid spins.')
        
        self.__func  = func
        self.__arrow = arrow
        self.__spins = spins
    
    def __getitem__(self,item):
        return self.__func(item[0],item[1])
    
    @property
    def arrow(self): return self.__arrow
    
    @property
    def spins(self): return self.__spins
    
    @property
    def dimension(self): return len(self.spins)
    
    def codomain(self,rank):
        return self.arrow + rank
    
    def ranks(self,rank):
        return (self.codomain(rank),rank)
    
    def shape(self,rank):
        return tuple(self.dimension**r for r in self.ranks(rank))
    
    def __call__(self,rank):
        r_out, r_in = self.ranks(rank)
        M = np.zeros(self.shape(rank))
        for i, sigma in enumerate(indices(r_out,indexing=self.spins)):
            for j, tau in enumerate(indices(r_in,indexing=self.spins)):
                M[i,j] = self[sigma,tau]
        return M
    
    @property
    def T(self):
        def func(sigma,tau):
            return self[tau,sigma]
        return SpinOperator(func,-self.arrow,self.spins)
    
    @array_check
    def __matmul__(self,other):
        def func(sigma,tau):
            K = indices(other.codomain(len(tau)),indexing=self.spins)
            
            return sum(self[sigma,kappa]*other[kappa,tau] for kappa in K)
        return SpinOperator(func,self.arrow+other.arrow,self.spins)
    
    @array_check
    def __add__(self,other):
        if self.arrow != other.arrow:
            raise TypeError('incompatible codomains')
        def func(sigma,tau):
            return self[sigma,tau] + other[sigma,tau]
        return SpinOperator(func,self.arrow,self.spins)
    
    @array_check
    def __mul__(self,other):
        def func(sigma,tau):
            return other*self[sigma,tau]
        return SpinOperator(func,self.arrow,self.spins)
    
    def __rmul__(self,other):
        return self*other
    
    def __truediv__(self,other):
        return self*(1/other)
        
    def __neg__(self):
        return (-1)*self
    
    def __pos__(self):
        return self
    
    def __sub__(self,other):
        return self + (-other)
    
    # right operations with np.ndarray
    def __array_ufunc__(self, *args):
        if args[0] == np.matmul:
            return (args[3].T @ args[2].T).T
        if args[0] == np.multiply:
            return args[3]*args[2]
        if args[0] == np.add:
            return args[3]+args[2]
        if args[0] == np.subtract:
            return -args[3]+args[2]
        pass

class Identity(SpinOperator):
    
    def __init__(self,**kwargs):
        SpinOperator.__init__(self,self.__identity,0,**kwargs)
    
    def __identity(self,sigma,tau):
        if sigma == tau:
            return 1
        return 0
    

class Metric(SpinOperator):
    
    def __init__(self,**kwargs):
        SpinOperator.__init__(self,self.__metric,0,**kwargs)
    
    def __metric(self,sigma,tau):
        if sigma == dual(tau):
            return 1
        return 0
    
    
class Transpose(SpinOperator):
    
    def __init__(self,permutation=(1,0),**kwargs):
        SpinOperator.__init__(self,self.__transpose,0,**kwargs)
        
        self.__permutation = permutation
        
    @property
    def permutation(self): return self.__permutation
    
    def __transpose(self,sigma,tau):
        if sigma == apply(self.permutation)(tau):
            return 1
        return 0
    
    
class Trace(SpinOperator):
    
    def __init__(self,indices=(0,1),**kwargs):
        SpinOperator.__init__(self,self.__trace,-len(indices),**kwargs)
    
        self.__indices = indices
        
    @property
    def indices(self): return self.__indices
    
    def __trace(self,sigma,tau):
        if sigma == remove(self.indices)(tau) and sum_(self.indices)(tau) == 0:
            return 1
        return 0
        

class Rotation(SpinOperator):
    
    def __init__(self,index=0,**kwargs):
        SpinOperator.__init__(self,self.__rotation,1,**kwargs)
        
        self.__index = index
        
    @property
    def index(self): return self.__index

    def __rotation(self,sigma,tau):
            return NotImplimented
    
class TensorProduct(SpinOperator):
    
    def __init__(self,element,**kwargs):
        if type(element) == int: element = (element,)
        SpinOperator.__init__(self,self.__product,len(element),**kwargs)
        
        self.__element = element
        
    @property
    def element(self): return self.__element

    def __product(self,sigma,tau):
        if sigma == self.element + tau:
            return 1
        return 0
