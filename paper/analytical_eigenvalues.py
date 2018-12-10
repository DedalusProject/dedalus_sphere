import numpy as np
from scipy.special import spherical_jn as j

def dispersion_zeros(ell,n,a=0,guess=None,imax=20,nk=10,eps=0.1):
    
    def F(k,deriv=False): 
        return j(ell,k,derivative=deriv) - a*j(ell+2,k,derivative=deriv)
    
    if guess == None:    
        kmax = np.pi*(n+ell/2 + eps)
        k = np.linspace(0,kmax,int(kmax*nk))
        S = np.sign(F(k))
        i = np.where(np.abs(np.roll(S,-1)-S)==2)[0]
        #print("You are getting %i roots. Deal with it." %(len(i)))
        k = 0.5*(k[i]+k[i+1])
    else: k = guess
    
    for i in range(imax):
        dk =  F(k)/F(k,deriv=True)
        k -= dk
    
    print('dk =',np.max(np.abs(dk)))
    
    return k


def wavenumbers(ell,n,BC):
    
    k = {'toroidal':0,'poloidal':0} 
    
    if BC=='Bessel':
        k = dispersion_zeros(ell,n)

    if BC=="no-slip":
        k['toroidal'] = dispersion_zeros(ell,n)
        k['poloidal'] = dispersion_zeros(ell+1,n)
        
    if BC=="stress-free":
        if ell == 1:
            k['toroidal'] = dispersion_zeros(2,n)
        else:
            k['toroidal'] = dispersion_zeros(ell-1,n,a=(ell+2)/(ell-1))
        k['poloidal'] = dispersion_zeros(ell,n,a=2/(2*ell+1))
        
    if BC=="potential":
        k['toroidal'] = dispersion_zeros(ell-1,n)
        k['poloidal'] = dispersion_zeros(ell,n)
        
    if BC=="conducting":
        k['toroidal'] = dispersion_zeros(ell,n)
        k['poloidal'] = dispersion_zeros(ell-1,n,a=ell/(ell+1))
        
    if BC=="pseudo":
        k['toroidal'] = dispersion_zeros(ell-1,n,a=ell/(ell+1))
        k['poloidal'] = dispersion_zeros(ell,n)
    
    return k 

def eigenvalues(k,n):
    kk = np.sort(np.concatenate((k['toroidal'],k['poloidal'])))
    kk = kk[0:min(n,len(kk))]**2
    return kk
