import numpy             as np

def xi(mu,ell):
    """
        Normalised derivative scale factors. xi(-1,ell)**2 + xi(+1,ell)**2 = 1.
        
        Parameters
        ----------
        mu  : int, regularity; -1,+1,0. xi(0,ell) = 0 by definition.
        ell : int spherical-harmonic degree.
        
        """
    
    return np.abs(mu)*np.sqrt((1 + mu/(2*ell+1))/2)

def forbidden_spin(ell,spin):
    if type(spin) == int: spin = [spin]
    return ell < abs(sum(spin))

def forbidden_regularity(ell,regularity):
    if type(regularity) == int: regularity = [regularity]
    
    walk = [ell]
    for r in regularity[::-1]:
        walk += [walk[-1] + r]
        if walk[-1] < 0 or ((walk[-1] == 0) and (walk[-2] == 0)): return True
    
    return False

def _swap(t,i,nu):
    L = list(t)
    L[i] = nu
    return tuple(L)

def regularity2spinMap(ell,spin,regularity):
    
    if spin == (): return 1
    
    if forbidden_spin(ell,spin) or forbidden_regularity(ell,regularity): return 0
    
    if type(spin) == int:
        rank = 1
        sigma, a = spin, regularity
        tau,   b = (), ()
    else:
        rank = len(spin)
        sigma, a = spin[0],  regularity[0]
        tau,   b = spin[1:], regularity[1:]
    
    R = 0
    for i in range(rank-1):
        if tau[i] == -sigma:
            R -= regularity2spinMap(ell,_swap(tau,i,0),b)
        if tau[i] == 0:
            R += regularity2spinMap(ell,_swap(tau,i,sigma),b)

    Qold   = regularity2spinMap(ell,tau,b)

    degree =  ell+sum(b)
    kangle = -sigma*np.sqrt((ell-sigma*sum(tau))*(ell+sigma*sum(tau)+1)/2)

    R -= kangle*Qold
    if sigma != 0: Qold = 0
    
    if a == -1: return (Qold*degree - R)/np.sqrt(degree*(2*degree+1))
    if a ==  0: return  sigma*R/np.sqrt(degree*(degree+1))
    if a == +1: return (Qold*(degree+1) + R)/np.sqrt((degree+1)*(2*degree+1))

def spin2regularityMap(ell,regularity,spin): return regularity2spinMap(ell,regularity,spin)

def tuple2index(tup,order=(-1,0,1)):
    index = 0
    for p,e in enumerate(tup[::-1]): index += (order[e+1]+1)*3**p
    return index

def index2tuple(index,rank,order=(-1,0,1)):
    
    tup = []
    while index > 0:
        
        tup = [order[index%3]] + tup
        index //= 3
    
    r = len(tup)
    if r < rank:
        tup = (rank-r)*[order[0]] + tup
    
    if r > rank: raise ValueError('rank smaller than tuple length.')
    
    return tuple(tup)

