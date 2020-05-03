
class Operator():
    
    def __init__(self,function,codomain):
    
        self.__function = function
        self.__codomain = codomain
    
    @property
    def codomain(self):
        return self.__codomain
    
    def __call__(self,*args):
        return self.__function(*args)
    
    def __matmul__(self,other):
        def function(*args):
            return self(*other.codomain(*args)) @ other(*args)
        return Operator(function, self.codomain + other.codomain)
    
    def __mul__(self,other):
        if type(other) == Operator:
            return self @ other - other @ self
        def func(*args):
            return other*self(*args)
        return Operator(func,self.codomain)
        
    def __add__(self,other):
        codomain = self.codomain | other.codomain
        def function(*args):
            return self(*args) + other(*args)
        return Operator(function, codomain)
    
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


