import numpy as np
from dedalus_sphere.ball128 import connection
from   scipy.linalg      import eig
import scipy.sparse      as sparse
import scipy.special     as spec

def BC_rows(N):
    N0 = N
    N1 = N + N0 + 1
    N2 = N + N1 + 1
    N3 = N + N2 + 1
    return N0,N1,N2,N3

def eigensystem(N,ell,B,alpha_BC,cutoff=1e9,boundary_conditions='no-slip'):

    if ell == 0: return 0, 0, 0, 0, 0, 0

    def D(mu,i,deg):
        if mu == +1: return B.op('D+',N,i,ell+deg)
        if mu == -1: return B.op('D-',N,i,ell+deg)

    def E(i,deg): return B.op('E',N,i,ell+deg)

    Z = B.op('0',N,0,ell)

    xim, xip = B.xi([-1,+1],ell)

    R00 = E(1,-1).dot(E( 0,-1))
    R11 = E(1, 0).dot(E( 0, 0))
    R22 = E(1,+1).dot(E( 0,+1))

    R=sparse.bmat([[R00, Z,   Z,  Z],
                   [Z, R11,   Z,  Z],
                   [Z,   Z, R22,  Z],
                   [Z,   Z,   Z,  Z]])
    R = R.tocsr()

    L00 = -D(-1,1, 0).dot(D(+1, 0,-1))
    L11 = -D(-1,1,+1).dot(D(+1, 0, 0))
    L22 = -D(+1,1, 0).dot(D(-1, 0,+1))

    L03 = xim*E(+1,-1).dot(D(-1,0,0))
    L23 = xip*E(+1,+1).dot(D(+1,0,0))

    L30 = xim*D(+1,0,-1)
    L32 = xip*D(-1,0,+1)

    L=sparse.bmat([[L00,  Z,   Z, L03],
                   [Z,  L11,   Z,   Z],
                   [Z,    Z, L22, L23],
                   [L30,  Z, L32,   Z]])

    N0, N1, N2, N3 = BC_rows(N)

    if boundary_conditions == 'no-slip':

        row0 = np.concatenate((               B.op('r=1',N,0,ell-1),np.zeros(N3-N0)))
        row1 = np.concatenate((np.zeros(N0+1),B.op('r=1',N,0,ell  ),np.zeros(N3-N1)))
        row2 = np.concatenate((np.zeros(N1+1),B.op('r=1',N,0,ell+1),np.zeros(N3-N2)))

    if boundary_conditions == 'stress-free':

        Q = B.Q[(ell,2)]
        rDmm = B.xi(-1,ell-1)*B.op('r=1',N,1,ell-2)*D(-1,0,-1)
        rDpm = B.xi(+1,ell-1)*B.op('r=1',N,1,ell  )*D(+1,0,-1)
        rDm0 = B.xi(-1,ell  )*B.op('r=1',N,1,ell-1)*D(-1,0, 0)
        rDp0 = B.xi(+1,ell  )*B.op('r=1',N,1,ell+1)*D(+1,0, 0)
        rDmp = B.xi(-1,ell+1)*B.op('r=1',N,1,ell  )*D(-1,0,+1)
        rDpp = B.xi(+1,ell+1)*B.op('r=1',N,1,ell+2)*D(+1,0,+1)

        rD = np.array([rDmm, rDm0, rDmp, 0.*rDmm, 0.*rDm0, 0.*rDmp, rDpm, rDp0, rDpp])
        QSm = Q[:,::3].dot(rD[::3])
        QS0 = Q[:,1::3].dot(rD[1::3])
        QSp = Q[:,2::3].dot(rD[2::3])
        u0m = B.op('r=1',N,0,ell-1)*B.Q[(ell,1)][1,0]
        u0p = B.op('r=1',N,0,ell+1)*B.Q[(ell,1)][1,2]

        row0=np.concatenate(( QSm[1]+QSm[3], QS0[1]+QS0[3], QSp[1]+QSp[3], np.zeros(N3-N2)))
        row1=np.concatenate(( u0m   ,       np.zeros(N0+1), u0p          , np.zeros(N3-N2)))
        row2=np.concatenate(( QSm[5]+QSm[7], QS0[5]+QS0[7], QSp[5]+QSp[7], np.zeros(N3-N2)))

    if boundary_conditions == 'potential-field':

        row0=np.concatenate((               B.op('r=1',N,0,ell-1),           np.zeros(N3-N0)))
        #L[N0]=np.concatenate((               B.op('r=1',N,1,ell  )*D(+1,0,-1),np.zeros(N3-N0)))
        row1=np.concatenate((np.zeros(N0+1),B.op('r=1',N,1,ell-1)*D(-1,0, 0),np.zeros(N3-N1)))
        row2=np.concatenate((np.zeros(N1+1),B.op('r=1',N,1,ell  )*D(-1,0,+1),np.zeros(N3-N2)))

    if boundary_conditions == 'pseudo-vacuum':

        row0=np.concatenate((               B.op('r=1',N,0,ell-1),           np.zeros(N3-N0)))
       #L[N0]=np.concatenate((               B.op('r=1',N,1,ell  )*D(+1,0,-1),np.zeros(N3-N0)))
        row1=np.concatenate((np.zeros(N0+1),B.op('r=1',N,1,ell-1)*D(-1,0, 0) - ell*B.op('r=1',N,0,ell),np.zeros(N3-N1)))
        row2=np.concatenate((np.zeros(N1+1),B.op('r=1',N,1,ell  )*D(-1,0,+1),np.zeros(N3-N2)))

    if boundary_conditions == 'perfectly-conducting':

        row0=np.concatenate((np.zeros(N2+1),B.op('r=1',N,0,ell)))
        row1=np.concatenate((np.zeros(N0+1),B.op('r=1',N,0,ell),np.zeros(N3-N1)))
        row2=np.concatenate((B.xi(+1,ell)*B.op('r=1',N,0,ell-1),np.zeros(N0+1),-B.xi(-1,ell)*B.op('r=1',N,0,ell+1),np.zeros(N3-N2)))


    C0 = connection(N,ell-1,alpha_BC,2)
    C1 = connection(N,ell  ,alpha_BC,2)
    C2 = connection(N,ell+1,alpha_BC,2)

    R00 = E(1,-1).dot(E( 0,-1))
    R11 = E(1, 0).dot(E( 0, 0))
    R22 = E(1,+1).dot(E( 0,+1))

    tau0 = C0[:,-1]
    tau0 = tau0.reshape((len(tau0),1))
    tau1 = C1[:,-1]
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = C2[:,-1]
    tau2 = tau2.reshape((len(tau2),1))

    col0 = np.concatenate((                   tau0,np.zeros((N3-N0,1))))
    col1 = np.concatenate((np.zeros((N0+1,1)),tau1,np.zeros((N3-N1,1))))
    col2 = np.concatenate((np.zeros((N1+1,1)),tau2,np.zeros((N3-N2,1))))

    L = sparse.bmat([[   L, col0, col1, col2],
                     [row0,    0 ,   0,    0],
                     [row1,    0 ,   0,    0],
                     [row2,    0,    0,    0]])

    R = sparse.bmat([[     R, 0*col0, 0*col1, 0*col2],
                     [0*row0,      0 ,     0,      0],
                     [0*row1,      0 ,     0,      0],
                     [0*row2,      0,      0,      0]])

    # The rate-limiting step
    vals, vecs = eig(L.todense(),b=R.todense())
    bad        = (np.abs(vals) > cutoff)
    vals[bad]  = np.inf
    good       = np.isfinite(vals)
    vals, vecs = vals[good], vecs[:,good]
    i          = np.argsort(vals.real)
    vals, vecs = vals[i], vecs.transpose()[i]

    return vals, vecs

