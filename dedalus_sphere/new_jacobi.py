import numpy             as np
from scipy.sparse import dia_matrix as banded

def increment_n(func):
    def wrapper(*args):
        return func(int(args[0]+1),*args[1:])
    wrapper.__name__ = func.__name__
    return wrapper


class Operator():
    
    def __init__(self,function,step):
    
        self.__func = function
        self.__step = step
    
    @property
    def step(self): return self.__step
    
    @property
    def func(self): return self.__func

    def __call__(self,*args):
        return self.func(*args)
    
    def __matmul__(self,other):
        
        def result(*args):
            brgs = np.array(args) + other.step
            return self.func(*brgs) @ other.func(*args)
            
        return Operator(result,self.step+other.step)
     
    def __mul__(self,other):
        
        if type(other) == Operator:
            return self @ other - other @ self
        
        if not type(other) in (int,float,complex):
            raise TypeError('Only scalar multiplication defined.')
 
        def result(*args): return other*self(*args)
        
        return Operator(result,self.step)
        
    def __rmul__(self,other):
            return self*other
    
    def __truediv__(self,other):
        return self * (1/other)
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return (-1)*self
    
    def __add__(self,other):
        
        if not (self.step == other.step).all():
            raise TypeError('Operators must have the same step.')
            
        def result(*args): return self(*args) + other(*args)
        
        return Operator(result,self.step)
    
    def __sub__(self,other):
        return self + (-other)
        
        
class JacobiOperator():
    
    def __init__(self,name):
        
        self.__func = {"A":self.__A,
                       "B":self.__B,
                       "C":self.__C,
                       "D":self.__D}[name]
        
    def __call__(self,p):
        p = int(p)
        if not p in (+1,-1):
            return ValueError('Must ladder by +1 or -1.')
            
        return Operator(*self.__func(p))
        
    def __A(self,p):
        
        @increment_n
        def A(n,a,b):

            N = np.arange(n)
            D = 2*N+a+b+1
            N = {+1:[N+(a+b+1),-(N+b)],
                 -1:[2*(N+a),-2*(N+1)]}[p]

            if a+b == -1: N[0][0], D[0] = 1/2, 1

            return banded((np.array(N)/D,[0,p]),(n+(1-p)//2,n))
        
        return A, np.array([(1-p)//2,p,0])

    def __B(self,p):
        
        @increment_n
        def B(n,a,b):
            
            N = np.arange(n)
            D = 2*N+a+b+1
            N = {+1:[N+(a+b+1),N+a],
                 -1:[2*(N+b),2*(N+1)]}[p]

            if a+b == -1: N[0][0], D[0] = 1/2, 1

            return banded((np.array(N)/D,[0,p]),(n+(1-p)//2,n))

        return B, np.array([(1-p)//2,0,p])
        
    def __C(self,p):
        
        @increment_n
        def C(n,a,b):
            
            N = [np.arange(n) + {+1:b,-1:a}[p]]

            return banded((N,[0]),(n,n))
        
        return C, np.array([0,p,-p])

    def __D(self,p):
        
        @increment_n
        def D(n,a,b):
            
            N = [(np.arange(n) + {+1:a+b+1,-1:1}[p])*2**(-p)]
        
            return banded((N,[p]),(n-p,n))
        
        return D, np.array([-p,p,p])
