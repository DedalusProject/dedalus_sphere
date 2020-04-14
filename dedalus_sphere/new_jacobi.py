import numpy             as np
import scipy.sparse      as sparse

default = lambda f: f
dense   = lambda f: f.todense()
banded  = sparse.dia_matrix
row     = sparse.csr_matrix
column  = sparse.csc_matrix

formatter = dense

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

def check_sign(func):
    def wrapper(self,p):
        if not p in (+1,-1): raise ValueError('Must ladder by +1 or -1.')
        return func(self,int(p))
    wrapper.__name__ = func.__name__
    return wrapper

def format(func):
    def wrapper(*args):
        return formatter(func(int(args[0]+1),*args[1:]))
    wrapper.__name__ = func.__name__
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

        return Operator(func,self.arrow+other.arrow)
    
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
        
        
class JacobiOperator():
    
    def __init__(self,name):
        
        self.__func = {"A":self.__A,
                       "B":self.__B,
                       "C":self.__C,
                       "D":self.__D}[name]
    
    @check_sign
    def __call__(self,p):
        return Operator(*self.__func(p))
        
    
    def __A(self,p):
        
        @format
        def A(n,a,b):

            N = np.arange(n)
            Q = 2*N+a+b+1
            N = {+1:[N+(a+b+1),-(N+b)],
                 -1:[2*(N+a),-2*(N+1)]}[p]

            if a+b == -1: N[0][0], Q[0] = 1/2, 1

            return banded((np.array(N)/Q,[0,p]),(n+(1-p)//2,n))
        
        return A, np.array([(1-p)//2,p,0])

    def __B(self,p):
        
        @format
        def B(n,a,b):
            
            N = np.arange(n)
            Q = 2*N+a+b+1
            N = {+1:[N+(a+b+1),N+a],
                 -1:[2*(N+b),2*(N+1)]}[p]

            if a+b == -1: N[0][0], Q[0] = 1/2, 1

            return banded((np.array(N)/Q,[0,p]),(n+(1-p)//2,n))

        return B, np.array([(1-p)//2,0,p])
        
    def __C(self,p):
        
        @format
        def C(n,a,b):
            
            N = [np.arange(n) + {+1:b,-1:a}[p]]

            return banded((N,[0]),(n,n))
        
        return C, np.array([0,p,-p])

    def __D(self,p):
        
        @format
        def D(n,a,b):
            
            N = [(np.arange(n) + {+1:a+b+1,-1:1}[p])*2**(-p)]
        
            return banded((N,[p]),(n-p,n))
        
        return D, np.array([-p,p,p])


class LaguerreOperator():

    def __init__(self,name):
        
        self.__func = {"A":self.__A,
                       "D":self.__D}[name]
        
    @check_sign
    def __call__(self,p):
        return Operator(*self.__func(p))
    
    def __A(self,p):
        
        @format
        def A(n,a):
            
            if p == +1:
                N = np.ones(n)
                N = [N,-N]
            if p == -1:
                N = np.arange(n)
                N = [N+a,-(N+1)]

            return banded((N,[0,p]),(n+(1-p)//2,n))
            
        return A, np.array([(1-p)//2,p])
        
    def __D(self,p):
        
        @format
        def D(n,a):

            if p == +1:
                N = [-np.ones(n)]
            if p == -1:
                N = [np.arange(n) + 1]

            return banded((N,[p]),(n-p,n))
            
        return D, np.array([-p,p])
        
        
class HermiteOperator():

    def __init__(self):
        
        self.__func = self.__D
        
    @check_sign
    def __call__(self,p):
        return Operator(*self.__func(p))
        
    def __D(self,p):

        @format
        def D(n):

            if p == +1:
                N = [2*np.arange(n)]
            if p == -1:
                N = [np.ones(n)]

            return banded((N,[p]),(n-p,n))
            
        return D, np.array([-p])
