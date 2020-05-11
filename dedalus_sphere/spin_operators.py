import numpy as np
from itertools import product
from operators import Operator

indexing = (-1,0,1)

# tuple helper functions
dual    =             lambda t: tuple(-e for e in t)
apply   = lambda p:   lambda t: tuple(t[int(i)] for i in p)
sum_    = lambda k:   lambda t: sum(t[int(i)] for i in k if 0 <= i < len(t))
remove  = lambda k:   lambda t: tuple(s for i,s in enumerate(t) if not i in k)
replace = lambda j,n: lambda t: tuple(s if i!=j else n for i,s in enumerate(t))

def int2tuple(func):
    if func.__name__ == '__getitem__':
        def wrapper(*args):
            self, args = args[0], args[1]
            if type(args) == int: return func(self,(args,))
            args = (self,tuple((s,) if type(s)==int else s for s in args))
            return func(*args)
        return wrapper
    return lambda *args: func(*tuple((s,) if type(s)==int else s for s in args))
        
def tuple2index(tup,indexing):
    return int('0'+''.join(str(indexing.index(s)) for s in tup),len(indexing))

def index2tuple(index,rank,indexing):
    s = np.base_repr(index,len(indexing),rank)
    return apply(s[(rank==0)-rank:])(indexing)


class TensorOperator(Operator):

    def __init__(self,function,codomain,indexing=indexing):
        Operator.__init__(self,function,codomain,Output=TensorOperator)
        self.__indexing = indexing
    
    @property
    def indexing(self):
        return self.__indexing
    
    @property
    def dimension(self):
        return len(self.indexing)
    
    @int2tuple
    def __getitem__(self,i):
        sigma,tau = i[0],i[1]
        i = tuple2index(sigma,self.indexing)
        j = tuple2index(tau,self.indexing)
        return self(len(tau))[i,j]
    
    def indices(self,rank):
        return product(*(rank*(self.indexing,)))
    
    def array(self,ranks):
        T = np.zeros(tuple(self.dimension**r for r in ranks))
        for i, sigma in enumerate(self.indices(ranks[0])):
            for j, tau in enumerate(self.indices(ranks[1])):
                T[i,j] = self[sigma,tau]
        return T

class Identity(TensorOperator):
    
    def __init__(self,indexing=indexing):
        
        identity = lambda rank: self.array((rank,rank))
        TensorOperator.__init__(self,identity,TensorCodomain(0),indexing=indexing)
        
    @int2tuple
    def __getitem__(self,i):
        return int(i[0] == i[1])


class Metric(TensorOperator):
    
    def __init__(self,indexing=indexing):
    
        metric = lambda rank: self.array((rank,rank))
        TensorOperator.__init__(self,metric,TensorCodomain(0),indexing=indexing)
    
    @int2tuple
    def __getitem__(self,i):
        return int(i[0] == dual(i[1]))
    
    
class Transpose(TensorOperator):
    
    def __init__(self,permutation=(1,0),indexing=indexing):
    
        transpose = lambda rank: self.array((rank,rank))
        TensorOperator.__init__(self,transpose,TensorCodomain(0),indexing=indexing)
        self.__permutation = permutation
    
    @property
    def permutation(self):
        return self.__permutation
    
    @int2tuple
    def __getitem__(self,i):
        return int(i[0] == apply(self.permutation)(i[1]))

class Trace(TensorOperator):
    
    def __init__(self,indices=(0,1),indexing=indexing):
    
        trace = lambda rank: self.array((rank-len(indices),rank))
        TensorOperator.__init__(self,trace,TensorCodomain(-len(indices)),indexing=indexing)
        
        self.__indices  = indices
    
    @property
    def indices(self):
        return self.__indices
    
    @int2tuple
    def __getitem__(self,i):
        return int(i[0] == remove(self.indices)(i[1]) and sum_(self.indices)(i[1]) == 0)
    
class TensorProduct(TensorOperator):
    
    def __init__(self,element,indexing=indexing):
        if type(element) == int: element = (element,)
        
        product = lambda rank: self.array((rank+len(element),rank))
        TensorOperator.__init__(self,product,TensorCodomain(len(element)),indexing=indexing)
        self.__element = element
    
    @property
    def element(self):
        return self.__element

    @int2tuple
    def __getitem__(self,i):
        return int(i[0] == self.element + i[1])
        


def xi(mu,ell):
    """
        Normalised derivative scale factors. xi(-1,ell)**2 + xi(+1,ell)**2 = 1.
        
        Parameters
        ----------
        mu  : int
            regularity; -1,+1,0. xi(0,ell) = 0 by definition.
        ell : int
            spherical-harmonic degree.
        
        """

    return np.abs(mu)*np.sqrt((1 + mu/(2*ell+1))/2)


class Intertwiner(TensorOperator):

    def __init__(self,L,indexing=indexing):
        
        intertwiner = lambda rank: self.array((rank,rank))
        TensorOperator.__init__(self,intertwiner,TensorCodomain(0),indexing=indexing)
        self.__ell = L

    @property
    def L(self):
        return self.__ell
        
    def k(self,s,mu):
        return -s*np.sqrt((self.L-s*mu)*(self.L+s*mu+1)/2)
    
    @int2tuple
    def forbidden_spin(self,spin):
        return self.L < abs(sum(spin))
    
    @int2tuple
    def forbidden_regularity(self,regularity):
        walk = (self.L,)
        for r in regularity[::-1]:
            walk += (walk[-1] + r,)
            if walk[-1] < 0 or walk[-2:] == (0,0):
                return True
        return False
    
    @int2tuple
    def __getitem__(self,i):
    
        spin, regularity = i[0], i[1]
        
        if spin == (): return 1

        if self.forbidden_spin(spin) or self.forbidden_regularity(regularity):
            return 0
        
        sigma, a = spin[0],  regularity[0]
        tau,   b = spin[1:], regularity[1:]
        
        R = 0
        for i,t in enumerate(tau):
            if t == -sigma: R -= self[replace(i,0)(tau),b]
            if t ==  0    : R += self[replace(i,sigma)(tau),b]

        Q  = self[tau,b]
        R -= self.k(sigma,sum(tau))*Q
        J  = self.L + sum(b)
        
        if sigma != 0: Q = 0
        
        if a == -1: return (Q * J - R)/np.sqrt(J*(2*J+1))
        if a ==  0: return  sigma*R/np.sqrt(J*(J+1))
        if a == +1: return (Q*(J+1) + R)/np.sqrt((J+1)*(2*J+1))
    

class TensorCodomain():

    def __init__(self,rank_change):
        self.__arrow = rank_change
        
    @property
    def arrow(self):
        return self.__arrow

    def __str__(self):
        s = f'(rank->rank+{self.arrow})'
        return s.replace('+0','').replace('+-','-')

    def __repr__(self):
        return str(self)
        
    def __add__(self,other):
        return TensorCodomain(self.arrow+other.arrow)

    def __call__(self,other):
        if self.arrow + other < 0:
            raise ValueError('cannot map to negative rank.')
        return (self.arrow + other,)
        
    def __eq__(self,other):
        return self.arrow == other.arrow
        
    def __or__(self,other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        return self
        
    def __neg__(self):
        return (-1)*self
        
    def __mul__(self,other):
        if type(other) != int:
            raise TypeError('only integer multiplication defined.')
        return TensorCodomain(other*self.arrow)

    def __rmul__(self,other):
        return self*other

    def __sub__(self,other):
        return self + (-other)
