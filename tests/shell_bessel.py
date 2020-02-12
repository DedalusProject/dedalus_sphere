from dedalus_sphere import annulus as shell
import numpy as np
from   scipy.linalg      import eig, inv
import scipy.special     as spec

# The eigenvalues should be the zeros of the half-integer-order Bessel functions; aka spherical bessel functions.

def eigensystem(N,l,radii,cutoff=1e9,report_error=False,alpha=(-0.5,-0.5),method='sparse',worland=False):

    z,w = shell.quadrature(N, alpha=alpha, niter=3,report_error=report_error)

    E0 = shell.operator(3, 'E',  N, 0, l, radii, alpha=alpha)
    E1 = shell.operator(3, 'E',  N, 1, l, radii, alpha=alpha)
    D0 = shell.operator(3, 'D+', N, 0, l, radii, alpha=alpha)
    D1 = shell.operator(3, 'D-', N, 1, l+1, radii, alpha=alpha)

    R = E1.dot(E0)
    L = D1.dot(D0)

    L,R = L.todense(), R.todense()

    L[-2] = shell.operator(3, 'r=Ri', N, 0, l, radii, alpha=alpha)
    L[-1] = shell.operator(3, 'r=Ro', N, 0, l, radii, alpha=alpha)
    R[-2] *= 0
    R[-1] *= 0

    vals, vecs = eig(L,b=-R)
    bad = (np.abs(vals) > cutoff)
    vals[bad] = np.nan
    vecs = vecs[:,np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]

    i = np.argsort(vals.real)
    vals, vecs = vals.real[i], vecs.real.transpose()[i]

    Ri, Ro = radii
    r = ( (Ro - Ri) * z + (Ro + Ri) )/2

    return vals, r, vecs

