import numpy             as np

def unitary(rank=1,adjoint=False):
    """ Transforms the components of vectors and tensors.
        U:        (v[th],v[ph]) --> (v[-],v[+])
        Uadjoint: (v[-],v[+])   --> (v[th],v[ph])
        
        Parameters
        ----------
        rank: int, rank=1 for vectors, rank=2 for matrices, etc
        adjoint: T/F returns the inverse transformation
        
        """
    
    if rank == 0: return np.array([[1]])
    
    U = np.sqrt(0.5)*np.array([[1,1],[1j,-1j]])
    
    if adjoint: U = U.conjugate().T
    
    unitary = U
    for k in range(rank-1):
        unitary = np.kron(U,unitary)
    
    return unitary


def spins(rank):
    spin = np.zeros(3**rank)
    for i in range(3**rank):
        spin[i] = _bar(i,rank)
    return spin


def make_maps(ell,rank):
    
    if rank == 0: return {0:np.array([[1]])}
    
    maps = make_maps(ell,rank-1)
    
    maps[rank] = _recurse_Map(maps[rank-1],ell,rank)
 
    return maps


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



def _k(mu,ell,s):
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

def _Q_normalization(ell,mu):
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

def _delta(mu,nu):
    if mu == nu: return 1.
    return 0.

def _get_element(nu,element_rank):
    nu = nu // 3**(element_rank)
    return (nu % 3) - 1

def _bar(mu,rank):
    mubar = 0
    for i in range(rank): mubar += _get_element(mu,i)
    return mubar

def _replace_index(nu,nup,i):
    """ nu_{i} -> nup """
    nui = _get_element(nu,i)
    nu -= (nui+1)*(3**i)
    return nu + (nup+1)*(3**i)

def _R(tau,mu,nu,Q,rank):
    R = 0
    if (rank == 0) or (mu == 0):
        return R
    for i in range(rank):
        nui = _get_element(nu,i)
        if (nui == +1 and mu == -1 ) or (nui == -1 and mu == +1 ) : R -= Q[_replace_index(nu,0,i),tau]
        elif (nui == 0 and mu == -1 ): R += Q[_replace_index(nu,-1,i),tau]
        elif (nui == 0 and mu == +1 ): R += Q[_replace_index(nu,+1,i),tau]
    return R

def _recurse_Map(Q_old,ell,new_rank):
    Q = np.zeros((3**new_rank,3**new_rank))
    for i in range(3**new_rank):
        for j in range(3**new_rank):
            mu,    nu  = (i//(3**(new_rank-1)))-1, i%(3**(new_rank-1))
            sigma, tau = (j//(3**(new_rank-1)))-1, j%(3**(new_rank-1))
            nubar      = _bar(nu,new_rank-1)
            deg, k_ang = ell + _bar(tau,new_rank-1), _k(mu,ell,nubar)
            Qnorm      = _Q_normalization(deg,sigma)
            S          = _R(tau,mu,nu,Q_old,new_rank-1)
            if Qnorm and (k_ang != np.inf):
                if   sigma == -1:
                    Q[i,j] = Qnorm*(    Q_old[nu,tau]*((deg  )*_delta(mu,0)+k_ang)-S)/(2*deg+1)
                elif sigma == 0:
                    Q[i,j] = Qnorm*(-mu*Q_old[nu,tau]*k_ang + mu*S )/(deg+1)
                elif sigma == 1:
                    Q[i,j] = Qnorm*(    Q_old[nu,tau]*((deg+1)*_delta(mu,0)-k_ang)+S)/(2*deg+1)
    return Q
