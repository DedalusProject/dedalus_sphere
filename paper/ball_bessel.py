from dedalus_sphere import ball128 as ball
import numpy as np
from   scipy.linalg      import eig, inv
import scipy.special     as spec

# The eigenvalues should be the zeros of the half-integer-order Bessel functions; aka spherical bessel functions.

def eigensystem(N,l,cutoff=1e9,report_error=False,a=0.0,method='sparse',worland=False):

    z,w = ball.quadrature(N,niter=3,report_error=report_error,a=a)

    Q = ball.polynomial(N,0,l,z,a=a)

    E0 = ball.operator('E',N,0,l,a=a)
    E1 = ball.operator('E',N,1,l,a=a)
    D0 = ball.operator('D+',N,0,l,a=a)
    D1 = ball.operator('D-',N,1,l+1,a=a)

    R = E1.dot(E0)
    L = D1.dot(D0)

    L,R = L.todense(), R.todense()
    L[N]=ball.operator('r=1',N,0,l,a=a)
    R[N]=np.zeros(N+1)

    vals, vecs = eig(L,b=-R)
    bad = (np.abs(vals) > cutoff)
    vals[bad] = np.nan
    vecs = vecs[:,np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]

    i = np.argsort(vals.real)
    vals, vecs = vals.real[i], vecs.real.transpose()[i]

    return vals, np.sqrt(0.5*(1+z)), vecs

