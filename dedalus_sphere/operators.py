from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import identity

class Operator():
    
    def __init__(self,function,codomain):
        
        self.__function = function
        self.__codomain = codomain
        
    @property
    def codomain(self):
        return self.__codomain
        
    def data(self,*args):
        return self(*args).A
    
    @property
    def T(self):
        codomain = -self.codomain
        def function(*args):
            return self(*codomain(*args)).T
        return Operator(function,codomain)
    
    
    def __call__(self,*args):
        return self.__function(*args)
    
    def __matmul__(self,other):
        def function(*args):
            return self(*other.codomain(*args)) @ other(*args)
        return Operator(function, self.codomain + other.codomain)
    
    def __add__(self,other):
        codomain = self.codomain | other.codomain
        def function(*args):
            return self(*args) + other(*args)
        return Operator(function, codomain)
    
    def __mul__(self,other):
        if type(other) == Operator:
            return self @ other - other @ self
        def function(*args):
            return other*self(*args)
        return Operator(function,self.codomain)
    
    def __rmul__(self,other):
            return self*other
    
    def __truediv__(self,other):
        return self * (1/other)
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return (-1)*self
    
    def __sub__(self,other):
        return self + (-other)


class infinite_csr(csr_matrix):

    def __init__(self,*args,**kwargs):
        csr_matrix.__init__(self,*args,**kwargs)
    
    @property
    def T(self):
        return infinite_csr(csr_matrix(self).T)
    
    def __add__(self,other):
        
        ns, no = self.shape[0], other.shape[0]
        
        if ns == no:
            sum_ = csr_matrix(self) + csr_matrix(other)
        
        if ns > no:
            sum_ = lil_matrix(self)
            sum_[:no] += other
            
        if ns < no:
            sum_ = lil_matrix(other)
            sum_[:ns] += self
        
        return infinite_csr(sum_)
