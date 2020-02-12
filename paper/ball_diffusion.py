import numpy as np
from dedalus_sphere import ball
from dedalus_sphere import intertwiner
from dedalus_sphere.jacobi128 import connection
from   scipy.linalg      import eig
import scipy.sparse      as sparse
import scipy.special     as spec

def BC_rows(N,ell,deg):
    N_list = []
    for d in deg:
        N_list.append( N - max((ell + d)//2,0) + 1 )
    if len(deg) == 1: return N_list
    N_list = np.cumsum(N_list)
    return N_list

def eigensystem(N, ell, alpha_BC, cutoff=1e9, boundary_conditions='no-slip'):

    if ell == 0: return 0, 0, 0, 0, 0, 0

    def D(mu,i,deg):
        if mu == +1: return ball.operator(3,'D+',N,i,ell,deg)
        if mu == -1: return ball.operator(3,'D-',N,i,ell,deg)

    def E(i,deg): return ball.operator(3,'E',N,i,ell,deg)

    def Z(deg_out,deg_in): return ball.zeros(N,ell,deg_out,deg_in)

    xim, xip = intertwiner.xi(np.array([-1,+1]),ell)

    R00 = E(1,-1).dot(E( 0,-1))
    R11 = E(1, 0).dot(E( 0, 0))
    R22 = E(1,+1).dot(E( 0,+1))

    Z01 = Z(-1, 0)
    Z02 = Z(-1,+1)
    Z03 = Z(-1, 0)
    Z10 = Z( 0,-1)
    Z12 = Z( 0,+1)
    Z13 = Z( 0, 0)
    Z20 = Z(+1,-1)
    Z21 = Z(+1, 0)
    Z23 = Z(+1, 0)
    Z30 = Z( 0,-1)
    Z31 = Z( 0, 0)
    Z32 = Z( 0,+1)
    Z33 = Z( 0, 0)

    R=sparse.bmat([[R00, Z01, Z02, Z03],
                   [Z10, R11, Z12, Z13],
                   [Z20, Z21, R22, Z23],
                   [Z30, Z31, Z32, Z33]])
    R = R.tocsr()

    L00 = -D(-1,1, 0).dot(D(+1, 0,-1))
    L11 = -D(-1,1,+1).dot(D(+1, 0, 0))
    L22 = -D(+1,1, 0).dot(D(-1, 0,+1))

    L03 = xim*E(+1,-1).dot(D(-1,0,0))
    L23 = xip*E(+1,+1).dot(D(+1,0,0))

    L30 = xim*D(+1,0,-1)
    L32 = xip*D(-1,0,+1)

    L=sparse.bmat([[L00, Z01, Z02, L03],
                   [Z10, L11, Z12, Z13],
                   [Z20, Z21, L22, L23],
                   [L30, Z31, L32, Z33]])

    N0, N1, N2, N3 = BC_rows(N,ell,[-1,0,+1,0])

    if boundary_conditions == 'no-slip':

        row0 = np.concatenate((               ball.operator(3,'r=R',N,0,ell,-1),np.zeros(N3-N0)))
        row1 = np.concatenate((np.zeros(N0),ball.operator(3,'r=R',N,0,ell,0),np.zeros(N3-N1)))
        row2 = np.concatenate((np.zeros(N1),ball.operator(3,'r=R',N,0,ell,+1),np.zeros(N3-N2)))

    if boundary_conditions == 'stress-free':

        Q1 = np.zeros((3,3))
        for j in range(3):
                for k in range(3):
                    s = intertwiner.index2tuple(j,1,indexing=(-1,0,+1))
                    r = intertwiner.index2tuple(k,1,indexing=(-1,0,+1))
                    Q1[j,k] = intertwiner.regularity2spinMap(ell,s,r)

        Q2 = np.zeros((9,9))
        for j in range(9):
                for k in range(9):
                    s = intertwiner.index2tuple(j,2,indexing=(-1,0,+1))
                    r = intertwiner.index2tuple(k,2,indexing=(-1,0,+1))
                    Q2[j,k] = intertwiner.regularity2spinMap(ell,s,r)

        rDmm = intertwiner.xi(-1,ell-1)*ball.operator(3,'r=R',N,1,ell,-2)*D(-1,0,-1)
        rDpm = intertwiner.xi(+1,ell-1)*ball.operator(3,'r=R',N,1,ell, 0)*D(+1,0,-1)
        rDm0 = intertwiner.xi(-1,ell  )*ball.operator(3,'r=R',N,1,ell,-1)*D(-1,0, 0)
        rDp0 = intertwiner.xi(+1,ell  )*ball.operator(3,'r=R',N,1,ell,+1)*D(+1,0, 0)
        rDmp = intertwiner.xi(-1,ell+1)*ball.operator(3,'r=R',N,1,ell, 0)*D(-1,0,+1)
        rDpp = intertwiner.xi(+1,ell+1)*ball.operator(3,'r=R',N,1,ell,+2)*D(+1,0,+1)

        rD = np.array([rDmm, rDm0, rDmp, 0.*rDmm, 0.*rDm0, 0.*rDmp, rDpm, rDp0, rDpp])
        QSm = Q2[:, ::3].dot(rD[::3])
        QS0 = Q2[:,1::3].dot(rD[1::3])
        QSp = Q2[:,2::3].dot(rD[2::3])
        u0m = ball.operator(3,'r=R',N,0,ell,-1)*Q1[1,0]
        u0p = ball.operator(3,'r=R',N,0,ell,+1)*Q1[1,2]

        row0=np.concatenate(( QSm[1]+QSm[3], QS0[1]+QS0[3], QSp[1]+QSp[3], np.zeros(N3-N2)))
        row1=np.concatenate(( u0m   ,      np.zeros(N1-N0), u0p          , np.zeros(N3-N2)))
        row2=np.concatenate(( QSm[5]+QSm[7], QS0[5]+QS0[7], QSp[5]+QSp[7], np.zeros(N3-N2)))

    if boundary_conditions == 'potential-field':

        row0=np.concatenate((             ball.operator(3,'r=R',N,0,ell,-1),           np.zeros(N3-N0)))
        row1=np.concatenate((np.zeros(N0),ball.operator(3,'r=R',N,1,ell,-1)*D(-1,0, 0),np.zeros(N3-N1)))
        row2=np.concatenate((np.zeros(N1),ball.operator(3,'r=R',N,1,ell, 0)*D(-1,0,+1),np.zeros(N3-N2)))

    if boundary_conditions == 'pseudo-vacuum':

        row0=np.concatenate((               ball.operator(3,'r=R',N,0,ell,-1),           np.zeros(N3-N0)))
        row1=np.concatenate((np.zeros(N0),ball.operator(3,'r=R',N,1,ell,-1)*D(-1,0, 0) - ell*ball.operator(3,'r=R',N,0,ell,0),np.zeros(N3-N1)))
        row2=np.concatenate((np.zeros(N1),ball.operator(3,'r=R',N,1,ell, 0)*D(-1,0,+1),np.zeros(N3-N2)))

    if boundary_conditions == 'perfectly-conducting':

        row0=np.concatenate((np.zeros(N2),ball.operator(3,'r=R',N,0,ell,0)))
        row1=np.concatenate((np.zeros(N0),ball.operator(3,'r=R',N,0,ell,0),np.zeros(N3-N1)))
        row2=np.concatenate((intertwiner.xi(+1,ell)*ball.operator(3,'r=R',N,0,ell,-1),
                             np.zeros(N1-N0),-intertwiner.xi(-1,ell)*ball.operator(3,'r=R',N,0,ell,+1),np.zeros(N3-N2)))

    ab = (alpha_BC,ell-1+0.5)
    cd = (2,       ell-1+0.5)
    C0 = connection(N - max((ell - 1)//2,0),ab,cd)
    ab = (alpha_BC,ell  +0.5)
    cd = (2,       ell  +0.5)
    C1 = connection(N - max((ell    )//2,0),ab,cd)
    ab = (alpha_BC,ell+1+0.5)
    cd = (2,       ell+1+0.5)
    C2 = connection(N - max((ell + 1)//2,0),ab,cd)

    tau0 = C0[:,-1]
    tau0 = tau0.reshape((len(tau0),1))
    tau1 = C1[:,-1]
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = C2[:,-1]
    tau2 = tau2.reshape((len(tau2),1))

    col0 = np.concatenate((                 tau0,np.zeros((N3-N0,1))))
    col1 = np.concatenate((np.zeros((N0,1)),tau1,np.zeros((N3-N1,1))))
    col2 = np.concatenate((np.zeros((N1,1)),tau2,np.zeros((N3-N2,1))))

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

