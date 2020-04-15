import numpy             as np
from itertools import product 

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

def _replace(t,i,nu):
    return tuple(nu if i==j else t[j] for j in range(len(t)))


def regularity2spinMap(ell,spin,regularity):
    
    if spin == (): return 1

    if forbidden_spin(ell,spin) or forbidden_regularity(ell,regularity): return 0
    
    if len(spin) != len(regularity):
        raise TypeError('spin and regularity must have the same length.')
    
    if type(spin) == int:
        order = 1
        sigma, a = spin, regularity
        tau,   b = (), ()
    else:
        order = len(spin)
        sigma, a = spin[0],  regularity[0]
        tau,   b = spin[1:], regularity[1:]

    R = 0
    for i in range(order-1):
        if tau[i] == -sigma:
            R -= regularity2spinMap(ell,_replace(tau,i,0),b)
        if tau[i] == 0:
            R += regularity2spinMap(ell,_replace(tau,i,sigma),b)

    Qold   = regularity2spinMap(ell,tau,b)

    degree =  ell+sum(b)
    kangle = -sigma*np.sqrt((ell-sigma*sum(tau))*(ell+sigma*sum(tau)+1)/2)

    R -= kangle*Qold
    if sigma != 0: Qold = 0

    if a == -1: return (Qold*degree - R)/np.sqrt(degree*(2*degree+1))
    if a ==  0: return  sigma*R/np.sqrt(degree*(degree+1))
    if a == +1: return (Qold*(degree+1) + R)/np.sqrt((degree+1)*(2*degree+1))

def spin2regularityMap(ell,regularity,spin):
    return regularity2spinMap(ell,regularity,spin)

def tuple2index(tup,indexing=(-1,1,0)):
    index = 0
    for p,e in enumerate(tup[::-1]): index += (indexing[e+1]+1)*3**p
    return index

def index2tuple(index,order,indexing=(-1,1,0)):

    tup = []
    while index > 0:

        tup = [indexing[index%3]] + tup
        index //= 3

    r = len(tup)
    if r < order:
        tup = (order-r)*[indexing[0]] + tup

    if r > order: raise ValueError('tensor order smaller than tuple length.')

    return tuple(tup)


class NCCCoupling():
    
    def __init__(self,ell,product_type):
        
        self.ell = ell
        
        self._func = {'SS' :self._S_T,
                      'V@V':self._V_dot_V,
                      'SV' :self._S_T,
                      'VxV':self._V_x_V,
                      'VS' :self._V_S,
                      'T@V':self._T_dot_V,
                      'ST' :self._S_T}[product_type]
        
                  
    def __call__(self,*abc):
        return self._func(*abc)

    def _Q3(self,sigma,tau,kappa,a,b,c):
            Q = regularity2spinMap
            return Q(self.ell,kappa,c)*Q(self.ell,tau,b)*Q(0,sigma,a)
        
    def _spins(self,rank):
        if rank == 1: return (-1,0,1)
        s = rank*((-1,0,1),)
        return product(*s)

    # scalar tensor/vector/scalar: (), (__,), (__,)
    def _S_T(self,*abc):
        if abc[2] == abc[1] and abc[0] == (): return 1
        return 0

    # vector dot vector:  (_,), (), (_,)
    def _V_dot_V(self,*abc):
        return sum(self._Q3((s,),(-s,),(),*abc) for s in self._spins(1))

    # vector scalar: (_,), (_,), ()
    def _V_S(self,*abc):
        return sum(self._Q3((s,),(),(s,),*abc) for s in self._spins(1))

    # vector cross vector: (_,), (_,), (_,)
    def _V_x_V(self,*abc):
        
        E = lambda sig, tau: 1j*np.roll(self._spins(1),sig)[tau+1]
            
        return sum(E(s,t)*self._Q3((s,),(t,),(s+t,),*abc) for s,t in self._spins(2))

    # tensor dot vector: (_,_), (_,) (_,)
    def _T_dot_V(self,*abc):
        return sum(self._Q3((t,-s),(s,),(t,),*abc) for s,t in self._spins(2))
        
