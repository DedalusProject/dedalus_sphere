import numpy             as np
from . import jacobi128  as jacobi

# The defalut configurations for the base Jacobi parameters.
alpha = 0

def quadrature(dimension,Nmax,alpha=alpha,**kw):
    return jacobi.quadrature(Nmax,alpha,dimension/2-1,**kw)

    #z, w = jacobi.quadrature(Nmax,alpha,1/2,**kw)
    #return z, w/np.sqrt(32), put this into volume integral


def trial_functions(dimension,Nmax,ell,degree,z,alpha=alpha):
    
    a, b, N = _regularity2Jacobi(dimension,Nmax,0,ell,degree,alpha=alpha)
    
    init = jacobi.envelope(a,b,a,dimension/2-1,z)
    return jacobi.recursion(N,a,b,z,init)


def operator(dimension,op,k,ell,degree,radius=1,pad=0,alpha=alpha):
    
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
    n = Nmax - max((ell + degree)//2,0)
    
    if ddeg == None: return a, b, n
    
    dn = max((ell + degree)//2,0) - max((ell + degree + ddeg)//2,0)
    
    return a, b, n, dn

def xi(mu,ell):
    """
        Normalised derivative scale factors. xi(-1,ell)**2 + xi(+1,ell)**2 = 1.

        Parameters
        ----------
        mu  : int
        ball spin parameter. Must be -1,+1,0. xi(0,l) = 0 by definition.
        ell : int
        spherical harmonic degree.

        """
    if mu == [-1,+1]: return xi(-1,ell), xi(+1,ell)
    if mu == [+1,-1]: return xi(+1,ell), xi(-1,ell)
    if mu == -1: return np.sqrt(ell/(2*ell+1))
    if mu == +1: return np.sqrt((ell+1)/(2*ell+1))
    return 0

def k(mu,ell,s):
    """
        Angular deravitive scale factors.

        Parameters
        ----------
        mu  : int
        ball spin parameter. Must be -1,+1,0. k(0,l,s) = 0 by definition.
        ell : int
        spherical harmonic degree.
        s   : int
        total spin weight

        """
    if (ell < mu*s) or (ell < -mu*s - 1): return np.inf
    return -mu*np.sqrt((ell-mu*s)*(ell+mu*s+1)/2)

def spins(rank):
    spin = np.zeros(3**rank)
    for i in range(3**rank):
        spin[i] = bar(i,rank)
    return spin

def Q_normalization(ell,mu):
    # returns the normalization of the Q matrix for ell > 0 or ell = 0 and mu = +1
    # otherwise returns None
    if ell > 0:
        if mu == 0:
            return np.sqrt( (ell+1)/ell )
        else:
            return 1/xi(mu,ell)
    elif ell == 0 and mu == 1:
        return 1.
    return None

def delta(mu,nu):
    if mu == nu: return 1.
    return 0.

def get_element(nu,element_rank):
    nu = nu // 3**(element_rank)
    return (nu % 3) - 1

def bar(mu,rank):
    mubar = 0
    for i in range(rank): mubar += get_element(mu,i)
    return mubar

def replace_index(nu,nup,i):
    """ nu_{i} -> nup """
    nui = get_element(nu,i)
    nu -= (nui+1)*(3**i)
    return nu + (nup+1)*(3**i)

def R(tau,mu,nu,Q,rank):
    R = 0
    if (rank == 0) or (mu == 0):
        return R
    for i in range(rank):
        nui = get_element(nu,i)
        if (nui == +1 and mu == -1 ) or (nui == -1 and mu == +1 ) : R -= Q[replace_index(nu,0,i),tau]
        elif (nui == 0 and mu == -1 ): R += Q[replace_index(nu,-1,i),tau]
        elif (nui == 0 and mu == +1 ): R += Q[replace_index(nu,+1,i),tau]
    return R

def recurseQ(Q_old,ell,rank):
    Q = np.zeros((3**rank,3**rank))
    for i in range(3**rank):
        for j in range(3**rank):
            mu,    nu  = (i//(3**(rank-1)))-1, i%(3**(rank-1))
            sigma, tau = (j//(3**(rank-1)))-1, j%(3**(rank-1))
            nubar      = bar(nu,rank-1)
            deg, k_ang = ell+bar(tau,rank-1), k(mu,ell,nubar)
            Qnorm      = Q_normalization(deg,sigma)
            S          = R(tau,mu,nu,Q_old,rank-1)
            if Qnorm and (k_ang != np.inf):
                if   sigma == -1:
                    Q[i,j] = Qnorm*(    Q_old[nu,tau]*((deg  )*delta(mu,0)+k_ang)-S)/(2*deg+1)
                elif sigma == 0:
                    Q[i,j] = Qnorm*(-mu*Q_old[nu,tau]*k_ang + mu*S )/(deg+1)
                elif sigma == 1:
                    Q[i,j] = Qnorm*(    Q_old[nu,tau]*((deg+1)*delta(mu,0)-k_ang)+S)/(2*deg+1)
    return Q
