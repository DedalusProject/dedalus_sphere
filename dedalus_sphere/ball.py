import numpy             as np
from . import jacobi128  as jacobi

# The defalut configurations for the base Jacobi parameter.
alpha = 0

def quadrature(dimension,Nmax,alpha=alpha,**kw):
    return jacobi.quadrature(Nmax,alpha,dimension/2-1,**kw)

    #z, w = jacobi.quadrature(Nmax,alpha,1/2,**kw)
    #return z, w/np.sqrt(32), put this into volume integral


def trial_functions(dimension,Nmax,ell,degree,z,alpha=alpha):
    
    a, b, N = _regularity2Jacobi(dimension,Nmax,0,ell,degree,alpha=alpha)
    
    init = jacobi.envelope(a,b,a,dimension/2-1,z)
    return jacobi.recursion(N,a,b,z,init)


def operator(dimension,op,Nmax,k,ell,degree,radius=1,pad=0,alpha=alpha):
    
    # derivatives, r multiplication
    if op in ['D-','D+','R-','R+']:
        
        ddeg = int(op[1]+'1')
        a, b, N, dN  = _regularity2Jacobi(dimension,Nmax+pad,k,ell,degree,ddeg=ddeg,alpha=alpha)

        if op == 'D-': op, rescale = 'C+', 2.0/radius
        if op == 'D+': op, rescale = 'D+', 2.0/radius
        if op == 'R-': op, rescale = 'B-', np.sqrt(0.5)*radius
        if op == 'R+': op, rescale = 'B+', np.sqrt(0.5)*radius
        
        if dN ==  0: return jacobi.operator(op,N  ,a,b,rescale=rescale)
        if dN ==  1: return jacobi.operator(op,N+1,a,b,rescale=rescale)[:,:-1]
        if dN == -1: return jacobi.operator(op,N  ,a,b,rescale=rescale)[:-1,:]
    

    a, b, N  = _regularity2Jacobi(dimension,Nmax+pad,k,ell,degree,alpha=alpha)
    
    # zeros
    if op == '0':  return jacobi.operator('0',N,a,b)
    
    # identity
    if op == 'I':  return jacobi.operator('I',N,a,b)

    # conversion
    if op == 'E':  return jacobi.operator('A+',N,a,b,rescale=np.sqrt(0.5))
    
    # z = 2*(r/R)**2 - 1 multiplication
    if op == 'Z': return jacobi.operator('J',N,a,b)
    
    if op == 'r=R': return jacobi.operator('z=+1',N,a,b,rescale=np.sqrt(2.0))


def _regularity2Jacobi(dimension,Nmax,k,ell,degree,ddeg=None,alpha=alpha):
    
    a = k + alpha
    b = ell + degree + dimension/2 - 1
    n = size(Nmax,ell,degree)
    
    if ddeg == None: return a, b, n
    
    dn = size(0,ell,degree+ddeg) - size(0,ell,degree)
    
    return a, b, n, dn

def size(Nmax,ell,degree): return Nmax - max((ell + degree)//2,0)

def zeros(Nmax, ell, deg_out, deg_in):
    Nout, Nin = size(Nmax,ell,deg_out), size(Nmax,ell,deg_in)
    return np.zeros((Nout+1,Nin+1),dtype=np.complex256)

