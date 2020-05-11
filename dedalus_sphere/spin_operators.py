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
    return lambda *args: int(func(*[(s,) if type(s)==int else s for s in args]))

def index2tuple(index,rank,indexing):
    s = np.base_repr(index,len(indexing),rank)
    return apply(s[(rank==0)-rank:])(indexing)
    
def tuple2index(tup,indexing):
    return int('0'+''.join(str(indexing.index(s)) for s in tup),len(indexing))

def indices(rank,indexing):
    return product(*(rank*(indexing,)))

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
    
    def __getitem__(self,i):
        sigma,tau = i[0],i[1]
        i = tuple2index(sigma,self.indexing)
        j = tuple2index(tau,self.indexing)
        return self(len(tau))[i,j]

    def array(self,ranks):
        T = np.zeros(tuple(self.dimension**r for r in ranks))
        for i, sigma in enumerate(indices(ranks[0],self.indexing)):
            for j, tau in enumerate(indices(ranks[1],self.indexing)):
                T[i,j] = self[sigma,tau]
        return T

class Identity(TensorOperator):
    
    def __init__(self,indexing=indexing):
        
        identity = lambda rank: self.array((rank,rank))
        TensorOperator.__init__(self,identity,TensorCodomain(0),indexing=indexing)
        
    @int2tuple
    def __getitem__(self,i):
        return i[0] == i[1]


class Metric(TensorOperator):
    
    def __init__(self,indexing=indexing):
    
        metric = lambda rank: self.array((rank,rank))
        TensorOperator.__init__(self,metric,TensorCodomain(0),indexing=indexing)
    
    @int2tuple
    def __getitem__(self,i):
        return i[0] == dual(i[1])
    
    
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
        return i[0] == apply(self.permutation)(i[1])

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
        return i[0] == remove(self.indices)(i[1]) and sum_(self.indices)(i[1]) == 0
    
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
