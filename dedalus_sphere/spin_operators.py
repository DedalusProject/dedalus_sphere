import numpy as np
from itertools import product
from itertools import permutations

from operators import Operator

indexing = (-1,0,1)

# tuple helper functions
dual    =             lambda t: tuple(-e for e in t)
apply   = lambda p:   lambda t: tuple(t[i] for i in p)
sum_    = lambda k:   lambda t: sum(t[i] for i in k if 0 <= i < len(t))
remove  = lambda k:   lambda t: tuple(s for i,s in enumerate(t) if not i in k)
replace = lambda j,n: lambda t: tuple(s if i!=j else n for i,s in enumerate(t))

def int2tuple(func):
    return lambda *args: int(func(*[(s,) if type(s)==int else s for s in args]))

def indices(rank,indexing):
    return product(*(rank*(indexing,)))

def tuple_array(elements,ranks,indexing):
    T = np.zeros(tuple(len(indexing)**r for r in ranks))
    for i, sigma in enumerate(indices(ranks[0],indexing)):
        for j, tau in enumerate(indices(ranks[1],indexing)):
            T[i,j] = elements[sigma,tau]
    return T


class Identity(Operator):
    
    def __init__(self,indexing=indexing):
        
        def identity(rank):
            return tuple_array(self,(rank,rank),indexing)
        
        Operator.__init__(self,identity,TensorCodomain(0))
        
    @int2tuple
    def __getitem__(self,i):
        return i[0] == i[1]


class Metric(Operator):
    
    def __init__(self,indexing=indexing):
    
        def metric(rank):
            return tuple_array(self,(rank,rank),indexing)
        
        Operator.__init__(self,metric,TensorCodomain(0))
    
    @int2tuple
    def __getitem__(self,i):
        return i[0] == dual(i[1])
    
    
class Transpose(Operator):
    
    def __init__(self,permutation=(1,0),indexing=indexing):
    
        def transpose(rank):
            return tuple_array(self,(rank,rank),indexing)
        
        Operator.__init__(self,transpose,TensorCodomain(0))
        self.__permutation = permutation
    
    @property
    def permutation(self):
        return self.__permutation
    
    @int2tuple
    def __getitem__(self,i):
        return i[0] == apply(self.permutation)(i[1])

class Trace(Operator):
    
    def __init__(self,indices=(0,1),indexing=indexing):
    
        def trace(rank):
            return tuple_array(self,(rank-len(indices),rank),indexing)
        
        Operator.__init__(self,trace,TensorCodomain(-len(indices)))
        
        self.__indices  = indices
    
    @property
    def indices(self):
        return self.__indices
    
    @int2tuple
    def __getitem__(self,i):
        return i[0] == remove(self.indices)(i[1]) and sum_(self.indices)(i[1]) == 0
    
class TensorProduct(Operator):
    
    def __init__(self,element,indexing=indexing):
        if type(element) == int: element = (element,)
        
        def product(rank):
            return tuple_array(self,(rank+len(element),rank),indexing)
        
        Operator.__init__(self,product,TensorCodomain(len(element)))
        self.__element = element
    
    @property
    def element(self):
        return self.__element

    @int2tuple
    def __getitem__(self,i):
        return i[0] == self.element + i[1]
    

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
