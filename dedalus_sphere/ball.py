import numpy             as np
from . import jacobi128  as jacobi

# The defalut configurations for the base Jacobi parameter.
alpha = 0

def quadrature(dimension,Nmax,alpha=alpha,**kw):
    """
           mass(a,b) = sum(weights)
      --> 2**(a+b+1) * factorial(a)*factorial(a)/factorial(a+b+1)
           
           if alpha == 0 and dimension == d:
               sum(weights) --> (1/d)*2**(d/2+1)
    """
    
    return jacobi.quadrature(Nmax,alpha,dimension/2-1,**kw)

def trial_functions(dimension,Nmax,ell,degree,z,alpha=alpha):


    a, b, N = _regularity2Jacobi_func(dimension,Nmax,0,ell,degree,alpha=alpha)

    init = jacobi.envelope(a,b,a,dimension/2-1,z)
    return jacobi.recursion(N,a,b,z,init)


def operator(dimension,op,Nmax,k,ell,degree,radius=1,pad=0,alpha=alpha):
     
    #    Ball : (Jacobi,rescale)
    O = {'D-' : ('C+',np.sqrt(8.0)/radius),   # d/dr + (ell+1)/r
         'D+' : ('D+',np.sqrt(8.0)/radius),   # d/dr -  ell/r
         'R-' : ('B-',np.sqrt(0.5)*radius),   # r multiplication
         'R+' : ('B+',np.sqrt(0.5)*radius),   # r multiplication
          'Z' : ('J',1),                      # z = 2*(r/R)**2 - 1 multiplication
          'I' : ('I',1),                      # identity
          'E' : ('A+',1),                     # conversion
          '0' : ('0',1),                      # zeros
        'r=R' : ('z=+1',2**((ell+degree)/2))} # boundary restriction
                                      
    a, b, N = _regularity2Jacobi_op(dimension,Nmax+pad,k,ell,degree,alpha=alpha)

    return jacobi.operator(O[op][0],N,a,b,rescale=O[op][1])

def _regularity2Jacobi_func(dimension,Nmax,k,ell,degree,alpha=alpha):

    a = k + alpha
    b = ell + degree + dimension/2 - 1
    n = Nmax - Nmin(ell,degree)

    return a, b, n

def _regularity2Jacobi_op(dimension,Nmax,k,ell,degree,alpha=alpha):

    a = k + alpha
    b = ell + degree + dimension/2 - 1
    n = Nmax - Nmin(ell,0)

    return a, b, n

def Nmin(ell,degree): return max((ell + degree)//2,0)

