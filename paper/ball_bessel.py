from dedalus_sphere import ball
import numpy as np
from   scipy.linalg      import eig, inv
import scipy.special     as spec

# The eigenvalues should be the zeros of the half-integer-order Bessel functions; aka spherical bessel functions.

def eigensystem(N,l,cutoff=1e9,report_error=False,alpha=0.0,method='sparse',worland=False):

    z,w = ball.quadrature(3, N, alpha=alpha, niter=3,report_error=report_error)

    E0 = ball.operator(3, 'E',  N, 0, l, 0, alpha=alpha)
    E1 = ball.operator(3, 'E',  N, 1, l, 0, alpha=alpha)
    D0 = ball.operator(3, 'D+', N, 0, l, 0, alpha=alpha)
    D1 = ball.operator(3, 'D-', N, 1, l, 1, alpha=alpha)

    R = E1.dot(E0)
    L = D1.dot(D0)

    L,R = L.todense(), R.todense()

    L[-1]=ball.operator(3, 'r=R', N, 0, l, 0, alpha=alpha)
    R[-1]*=0

    vals, vecs = eig(L,b=-R)
    bad = (np.abs(vals) > cutoff)
    vals[bad] = np.nan
    vecs = vecs[:,np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]

    i = np.argsort(vals.real)
    vals, vecs = vals.real[i], vecs.real.transpose()[i]

    return vals, np.sqrt(0.5*(1+z)), vecs

