from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import identity as id_matrix


class Operator():
    """
    Class for deffered (lazy) evaluation of matrix-valued functions between parameterised vector spaces.
    
    Over a set of possible vector spaces D = {domains},
    
        A: domain in D --> codomain(A)(domain) in D
        B: domain in D --> codomain(B)(domain) in D
    
        A @ B : domain --> codomain(B)(domain) --> codomain(AB)(domain).
    
    Operator strings are lazily evaluated on a given domin,
    
        (A @ B)(domain) = A(codomain(B)(domain)) @ B(domain).
    
    The codomains have a composition rule:
    
        codomain(A) + codomain(B) = codomain(AB).
    
    The composition rule need not be commutative, but it often is.
    
    Operators with compatible codomains form a linear vector space.

        a*A + b*B : domain in D --> codomain(A|B)(domain) in D,
        
    where codomain(A|B) = codomain(A) or codomain(B), provided they are compatible.
    
    For a given operator, we can define the inverse codomain such that,
        
        codomain(A)(domain)  + (-codomain(A)(domain)) = domain.
    
    This leads to the notion of a transpose operator,
        
        A.T : domain --> -codomain(A)(domain).
        
    and A @ A.T , A.T @ A : domain --> domain.
    
    The specific form of the transpose is given by A(domain).T for each domain.
        
        
    Attributes
    ----------
    codomain: an arrow between any given domain and codomain(domain).
    identity: The identity operator with the same type as self.
    Output  : class to cast output into. It should be a subclass of Operator.
    
    Methods
    -------
    self.data(*args):
        view of the matrix for given domain args.
    self(*args):
        evaluation of an operator object on domain args.
    self.T:
        returns transpose operator.
    self@other:
        operator composition.
    self+other:
        compatible operator addition.
    self*other:
        if self and other are both operators, retrurn the commutator A@B - B@A.
        Otherwise returns scalar multiplication.
    self**n: repeated composition.
    
    """
    
    def __init__(self,function,codomain,Output=None):
        if Output == None: Output = Operator
        
        self.__function = function
        self.__codomain = codomain
        self.__Output   = Output
        
    @property
    def function(self):
        return self.__function
    
    @property
    def codomain(self):
        return self.__codomain
    
    @property
    def Output(self):
        return self.__Output
        
    def data(self,*args):
        return self(*args).toarray()
    
    def __call__(self,*args):
        return self.__function(*args)
    
    def __matmul__(self,other):
        def function(*args):
            return self(*other.codomain(*args)) @ other(*args)
        return self.Output(function, self.codomain + other.codomain)
    
    @property
    def T(self):
        codomain = -self.codomain
        def function(*args):
            return self(*codomain(*args)).T
        return self.Output(function,codomain)
    
    @property
    def identity(self):
        def function(*args):
            return self(*args).identity
        return  self.Output(function,0*self.codomain)
        
    def __pow__(self,exponent):
        if exponent < 0:
            return TypeError('exponent must be a non-negative integer.')
        if exponent == 0:
            return self.identity
        return self @ self**(exponent-1)
    
    def __add__(self,other):
        if other == 0: return self
        if not isinstance(other,Operator):
            other = other*self.identity
        codomain = self.codomain | other.codomain
        if codomain is self.codomain:
            output = self.Output
        else:
            output = other.Output
        def function(*args):
            return self(*args) + other(*args)
        return output(function, codomain)
    
    def __mul__(self,other):
        if isinstance(other,Operator):
            return self @ other - other @ self
        def function(*args):
            return other*self(*args)
        return self.Output(function,self.codomain)
    
    def __radd__(self,other):
        return self + other
    
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
        
    def __rsub__(self,other):
        return -self + other
        

class infinite_csr(csr_matrix):
    """
    Base class for extendable addition with csr_matrix types.
    
    If A.shape = (j,n), and B.shape = (k,n) we can add A+B by only summing rows i <= min(j,k).
    This is equivalent to padding the small array with rows of zeros.
    
    The class inherits from csr_matrix.
    
    Attributes
    ----------
    self.T: transpose.
        csr_matrix traspose returns csc_matrix type.
    self.identity:
        returns the identity matrix with the same number of columns.
        
    Methods
    -------
    self + other: row-extendable addition.
    
    """

    def __init__(self,*args,**kwargs):
        csr_matrix.__init__(self,*args,**kwargs)
    
    @property
    def T(self):
        return infinite_csr(csr_matrix(self).T)
    
    @property
    def identity(self):
        return infinite_csr(id_matrix(self.shape[1]))
    
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
        
    def __radd__(self,other):
        return self + other
