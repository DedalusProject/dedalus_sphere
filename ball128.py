import numpy             as np
import scipy             as sp
import jacobi128         as jacobi
import scipy.sparse      as sparse

alpha = -0.5 # default for a default. Base value of floating Jabobi type index; could go in a configuration file.

def quadrature(N_max,a=alpha,**kw):
    z, w = jacobi.quadrature(N_max,a,1/2,**kw)
    return z, w/np.sqrt(32)

def polynomial(N,k,ell,z,a=alpha):
    """ N is max degree of output Jacobi polynomial. 
        There will be blood if N >> len(z) - ell//2."""
    
    q, m = k + a, ell+1/2

    init = np.sqrt(2**(k+5/2))*jacobi.envelope(q,m,q,1/2,z)
    
    return jacobi.recursion(N,q,m,z,init)

def N_min(ell):
    """ Minimum something degree. """
    return max(ell//2,0)

def connection(N,ell,a,b):
    """The connection matrix between any bases coefficients:

        Qa(r) = Qb(r) . C    -->    Fb = C . Fa

        C(a,b) = inverse(C(b,a))

        if b = a    --> C =         operator('I',N,0,ell,a)
        if b = a+1  --> C = sqrt(2)*operator('E',N,0,ell,a)

        The output is always a dense matrix format.

        Parameters
        ----------
        N, ell: int
        a > -1 :  input basis
        b > -1 : output basis

        """

    z, w = quadrature(N+ell//2,b)

    Qa = polynomial(N,0,ell,z,a)
    Qb = polynomial(N,0,ell,z,b)

    return Qb.dot((w*Qa).T)

def operator(op,N,k,ell,a=alpha):
    
    q, m = k + a, ell+1/2
    
    # zeros
    if (op == '0'):  return jacobi.operator('0',N,q,m)
    
    # identity
    if (op == 'I'):  return jacobi.operator('I',N,q,m)
    
    # conversion
    if (op == 'E'):  return jacobi.operator('A+',N,q,m,rescale=np.sqrt(0.5))
    
    # derivatives
    if (op == 'D-'): return jacobi.operator('C+',N,q,m,rescale=2.0)
    if (op == 'D+'): return jacobi.operator('D+',N,q,m,rescale=2.0)
    
    # r multiplication
    if (op == 'R-'): return jacobi.operator('B-',N,q,m,rescale=np.sqrt(0.5))
    if (op == 'R+'): return jacobi.operator('B+',N,q,m,rescale=np.sqrt(0.5))

    # z = 2*r*r-1 multiplication
    if op == 'Z': return jacobi.operator('J',N,q,m)
    
    if op == 'r=1': return jacobi.operator('z=+1',N,q,m,rescale=np.sqrt(2.0))

def zeros(N,ell,deg_out,deg_in):
    """ non-square array of zeros.""" # Cannot make an operator because of non-square.
    return sparse.csr_matrix((N+1-N_min(ell+deg_out),N+1-N_min(ell+deg_in)))

def unitary3D(rank=1,adjoint=False):
    """ Transforms the components of vectors and tensors.
        U:        (v[r],v[th],v[ph]) --> (v[-],v[0], v[+])
        Uadjoint: (v[-],v[0], v[+])  --> (v[r],v[th],v[ph])
        
        Parameters
        ----------
        rank: int
        rank=1 for vectors, rank=2 for matrices, etc
        adjoint: T/F
        returns the inverse transformation
        
        """
    
    if adjoint :
        U       = np.sqrt(0.5)*np.array([[0,1,-1j],[np.sqrt(2.0),0,0],[0,1,1j]])
    else:
        U       = np.sqrt(0.5)*np.array([[0,np.sqrt(2.0),0],[1,0,1],[1j,0,-1j]])

    unitary = U
    for k in range(rank-1):
        unitary = np.kron(U,unitary)
    return unitary

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

