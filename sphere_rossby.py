import sphere128 as sphere
import numpy as np
import scipy.sparse      as sparse
from   scipy.linalg      import eig

def eigensystem(Lmax,m,cutoff=1e9,report_error=False):
    # Rossby-Haurwitz waves. These work for abs(m) > 0, and give the correct gauge modes for m=0.
    # The eigenvlaues are the frequencies omega = m/(l*(l+1)) 
    
    e = np.sqrt(0.5)
    
    z,w = sphere.quadrature(Lmax,niter=3,report_error=report_error)

    Y = {'0':sphere.Y(Lmax,m, 0,z)}
    Y['+'] = sphere.Y(Lmax,m,+1,z)
    Y['-'] = sphere.Y(Lmax,m,-1,z)

    L00 = -1j*sphere.operator('C',Lmax,m,+1)
    L01 =         sphere.zeros(Lmax,m,1,-1)
    L02 =    sphere.operator('k+',Lmax,m,0)
    
    L10 =         sphere.zeros(Lmax,m,-1,+1)
    L11 =  1j*sphere.operator('C',Lmax,m,-1)
    L12 =    sphere.operator('k-',Lmax,m,0)
    
    L20 = sphere.operator('k-',Lmax,m,+1)
    L21 = sphere.operator('k+',Lmax,m,-1)
    L22 =        sphere.zeros(Lmax,m,0,0)

    L = sparse.bmat([[L00,L10,L20],[L01,L11,L21],[L02,L12,L22]])

    R00 = 1j*sphere.operator('I',Lmax,m,1)
    R01 = sphere.zeros(Lmax,m,1,-1)
    R02 = sphere.zeros(Lmax,m,1,0)

    R10 = sphere.zeros(Lmax,m,-1,1)
    R11 = 1j*sphere.operator('I',Lmax,m,-1)
    R12 = sphere.zeros(Lmax,m,-1,0)

    R20 = sphere.zeros(Lmax,m,0,1)
    R21 = sphere.zeros(Lmax,m,0,-1)
    R22 = sphere.zeros(Lmax,m,0,0)

    R = sparse.bmat([[R00,R10,R20],[R01,R11,R21],[R02,R12,R22]])

    L,R = L.todense(), R.todense()
    
    vals, vecs = eig(L,b=-R)
    bad = (np.abs(vals) > cutoff)
    vals[bad] = np.nan
    vecs = vecs[:,np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]
    
    i = np.argsort(-np.abs(vals))
    vals, vecs = vals[i], vecs.transpose()[i]
    
    N0 = Lmax - sphere.L_min(m,+1) + 1
    N1 = Lmax - sphere.L_min(m,-1) + 1 + N0
    N2 = Lmax - sphere.L_min(m, 0) + 1 + N1

    vth = e*(vecs[:,0:N0].dot(Y['+']) + vecs[:,N0:N1].dot(Y['-']))
    vph = e*(vecs[:,0:N0].dot(Y['+']) - vecs[:,N0:N1].dot(Y['-']))
    p   = vecs[:,N1:N2].dot(Y['0'])

    return vals, np.arccos(z), vth, vph, p



