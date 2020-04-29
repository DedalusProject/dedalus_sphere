import numpy             as np
import scipy.sparse      as sparse

def check_coefficient(multiply):
    def wrapper(self,other):
        if type(other) == Operator:
            return self @ other - other @ self
        if not type(other) in (int,float,complex):
            raise TypeError('Only scalar multiplication defined.')
        return multiply(self,other)
    wrapper.__name__ = multiply.__name__
    return wrapper

def check_codomain(add):
    def wrapper(self,other):
        if not (self.indices == other.indices).all():
            raise TypeError('Operators must have compatible codomains.')
        return add(self,other)
    wrapper.__name__ = add.__name__
    return wrapper


class Operator():
    
    def __init__(self,function,arrow):
    
        self.__func  = function
        self.__arrow = arrow
    
    @property
    def arrow(self): return self.__arrow
    
    @property
    def degree(self): return self.arrow[0]
    
    @property
    def indices(self): return self.arrow[1:]
    
    def codomain(self,*args):
        return tuple(np.array(args) + np.array(self.arrow))
    
    def __call__(self,*args):
        return self.__func(*args)
    
    def __matmul__(self,other):
        def func(*args):
            return self(*other.codomain(*args)) @ other(*args)
        return Operator(func,self.arrow + other.arrow)
    
    @check_coefficient
    def __mul__(self,other):
        def func(*args):
            return other*self(*args)
            
        return Operator(func,self.arrow)
        
    @check_codomain
    def __add__(self,other):
        def func(*args):
            return self.__embedded_sum(self(*args),other(*args))
        arrow = self.arrow
        arrow[0] = max(self.degree,other.degree)
        return Operator(func,arrow)
    
    def __embedded_sum(self,*M):
        n = (M[0].shape[0], M[1].shape[0])
        if  n[0] == n[1]: return M[0] + M[1]
        i = n.index(max(n))
        M[i][:n[1-i]] = M[i][:n[1-i]] + M[1-i][:n[1-i]]
        return M[i]
            
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
