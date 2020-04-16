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
    
    def int2tuple(func):
        return lambda *args: func(*[(s,) if type(s)==int else s for s in args])
    
    def __init__(self,ell,product_type):
        
        self.ell = ell
        
        self.__func = {'SS' :self.__S_T,
                       'V@V':self.__V_dot_V,
                       'SV' :self.__S_T,
                       'VxV':self.__V_x_V,
                       'VS' :self.__V_S,
                       'T@V':self.__T_dot_V,
                       'V@T':self.__V_dot_T,
                       'ST' :self.__S_T,
                       'TS' :self.__T_S,
                       'VV' :self.__V_V,
                       'T@T':self.__T_dot_T}[product_type]
    
    
    def __call__(self,*abc):
        if self.selection_rule(*abc):
            return self.__func(*abc)
        return 0
        
    @int2tuple
    def selection_rule(self,*abc):
        a,b,c = tuple(map(sum,abc))
        d = a-abs(c-b)
        return (d >= 0) and (d % 2 == 0)
        
    def _spins(self,rank):
        if rank == 1: return (-1,0,1)
        s = rank*((-1,0,1),)
        return product(*s)
    
    @int2tuple
    def __Q3(self,sigma,tau,kappa,a,b,c):
        Q = regularity2spinMap
        return Q(self.ell,kappa,c)*Q(self.ell,tau,b)*Q(0,sigma,a)
    
    # scalar tensor/vector/scalar
    @int2tuple
    def __S_T(self,*abc):
        if abc[0] == () and abc[1] == abc[2] : return 1
        return 0

    # vector dot vector
    @int2tuple
    def __V_dot_V(self,*abc):
        return self.__Q3(0,0,(),*abc)
        
    # vector scalar
    @int2tuple
    def __V_S(self,*abc):
        return self.__Q3(0,(),0,*abc)
        
    # vector cross vector
    @int2tuple
    def __V_x_V(self,*abc):
        return 1j*(self.__Q3(0,+1,+1,*abc) - self.__Q3(0,-1,-1,*abc))
        
    # tensor dot vector
    @int2tuple
    def __T_dot_V(self,*abc):
        return sum(self.__Q3((s,-s),s,s,*abc) for s in self._spins(1))
        
    # vector dot tensor
    @int2tuple
    def __V_dot_T(self,*abc):
        return sum(self.__Q3(0,(0,s),s,*abc) for s in self._spins(1))
        
    # tensor scalar
    @int2tuple
    def __T_S(self,*abc):
        return sum(self.__Q3((s,-s),(),(s,-s),*abc) for s in self._spins(1))
        
    # vector vector
    @int2tuple
    def __V_V(self,*abc):
        return sum(self.__Q3(0,s,(0,s),*abc) for s in self._spins(1))
        
    # tensor dot tensor
    @int2tuple
    def __T_dot_T(self,*abc):
        return sum(self.__Q3((s,-s),(s,t),(s,t),*abc) for s,t in self._spins(2))
    
    
