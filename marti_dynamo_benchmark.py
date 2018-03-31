import ball_wrapper as ball
import ball128
import numpy as np
from   scipy.linalg      import eig
from scipy.sparse        import linalg as spla
import scipy.sparse      as sparse
import scipy.special     as spec
import dedalus.public as de
from dedalus.core.distributor import Distributor
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import timesteppers
import pickle


# Gives LHS matrices for boussinesq

def BC_rows(N):
    N0 = N
    N1 = N + N0 + 1
    N2 = N + N1 + 1
    N3 = N + N2 + 1
    N4 = N + N3 + 1
    N5 = N + N4 + 1
    N6 = N + N5 + 1
    N7 = N + N6 + 1
    N8 = N + N7 + 1
    return N0,N1,N2,N4,N5,N6,N7

def matrices(N,ell,Ekman,Rossby,q):
    
    def D(mu,i,deg):
        if mu == +1: return B.op('D+',N,i,ell+deg)
        if mu == -1: return B.op('D-',N,i,ell+deg)
    
    def E(i,deg): return B.op('E',N,i,ell+deg)
    
    def C(deg): return ball128.connection(N,ell+deg,alpha_BC,2)

    Z = B.op('0',N,0,ell)
    
    N0 = N
    N1 = N + N0 + 1
    N2 = N + N1 + 1
    N3 = N + N2 + 1
    N4 = N + N3 + 1
    N5 = N + N4 + 1
    N6 = N + N5 + 1
    N7 = N + N6 + 1
    N8 = N + N7 + 1
   
    if ell == 0:
        I = B.op('I',N,0,ell).tocsr()
        M = sparse.bmat([[Z,Z,Z,Z,Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,E(1, 0).dot(E( 0, 0)),Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z,Z,Z,Z,Z]]).tocsr()
        M[N4] = np.zeros(N8+1)
        L = sparse.bmat([[I,Z,Z,Z,Z,Z,Z,Z,Z],
                         [Z,I,Z,Z,Z,Z,Z,Z,Z],
                         [Z,Z,I,Z,Z,Z,Z,Z,Z],
                         [Z,Z,Z,I,Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,-q*D(-1,1,+1).dot(D(+1, 0, 0)),Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z,I,Z,Z,Z],
                         [Z,Z,Z,Z,Z,Z,I,Z,Z],
                         [Z,Z,Z,Z,Z,Z,Z,I,Z],
                         [Z,Z,Z,Z,Z,Z,Z,Z,I]]).tocsr()
        row0 = np.concatenate(( np.zeros(N3+1), B.op('r=1',N,0,ell), np.zeros(N8-N4) ))

        tau0 = (C(0))[:,-1]
        tau0 = tau0.reshape((len(tau0),1))

        col0 = np.concatenate((np.zeros((N3+1,1)),tau0,np.zeros((N8-N4,1))))

        L = sparse.bmat([[   L, col0],
                         [row0,    0]])

        M = sparse.bmat([[     M, 0*col0],
                         [0*row0,      0]])

        L = L.tocsr()
        M = M.tocsr()

        return M, L
    
    xim, xip = B.xi([-1,+1],ell)
    
    M00 = Rossby*E(1,-1).dot(E( 0,-1))
    M11 = Rossby*E(1, 0).dot(E( 0, 0))
    M22 = Rossby*E(1,+1).dot(E( 0,+1))
    M44 = E(1, 0).dot(E( 0, 0))
    M55 = E(1,-1).dot(E( 0,-1))
    M66 = E(1, 0).dot(E( 0, 0))
    M77 = E(1,+1).dot(E( 0,+1))

    M=sparse.bmat([[M00, Z,   Z,  Z,   Z,   Z,   Z,   Z,   Z],
                   [Z, M11,   Z,  Z,   Z,   Z,   Z,   Z,   Z],
                   [Z,   Z, M22,  Z,   Z,   Z,   Z,   Z,   Z],
                   [Z,   Z,   Z,  Z,   Z,   Z,   Z,   Z,   Z],
                   [Z,   Z,   Z,  Z, M44,   Z,   Z,   Z,   Z],
                   [Z,   Z,   Z,  Z,   Z, M55,   Z,   Z,   Z],
                   [Z,   Z,   Z,  Z,   Z,   Z, M66,   Z,   Z],
                   [Z,   Z,   Z,  Z,   Z,   Z,   Z, M77,   Z],
                   [Z,   Z,   Z,  Z,   Z,   Z,   Z,   Z,   Z]])
    M = M.tocsr()
                   
    L00 = -Ekman*D(-1,1, 0).dot(D(+1, 0,-1))
    L11 = -Ekman*D(-1,1,+1).dot(D(+1, 0, 0))
    L22 = -Ekman*D(+1,1, 0).dot(D(-1, 0,+1))
    L44 = -q*D(-1,1,+1).dot(D(+1, 0, 0))
    L55 = -D(-1,1, 0).dot(D(+1, 0,-1))
    L66 = -D(-1,1,+1).dot(D(+1, 0, 0))
    L77 = -D(+1,1, 0).dot(D(-1, 0,+1))
               
    L03 = L58 = xim*E(+1,-1).dot(D(-1,0,0))
    L23 = L78 = xip*E(+1,+1).dot(D(+1,0,0))
    
    L30 = L85 = xim*D(+1,0,-1)
    L32 = L87 = xip*D(-1,0,+1)

    L=sparse.bmat([[L00,  Z,   Z, L03,   Z,   Z,   Z,   Z,   Z],
                   [Z,  L11,   Z,   Z,   Z,   Z,   Z,   Z,   Z],
                   [Z,    Z, L22, L23,   Z,   Z,   Z,   Z,   Z],
                   [L30,  Z, L32,   Z,   Z,   Z,   Z,   Z,   Z],
                   [Z,    Z,   Z,   Z, L44,   Z,   Z,   Z,   Z],
                   [Z,    Z,   Z,   Z,   Z, L55,   Z,   Z, L58],
                   [Z,    Z,   Z,   Z,   Z,   Z, L66,   Z,   Z],
                   [Z,    Z,   Z,   Z,   Z,   Z,   Z, L77, L78],
                   [Z,    Z,   Z,   Z,   Z, L85,   Z, L87,   Z]])
    L = L.tocsr()

    Q = B.Q[(ell,2)]
    if ell == 1: rDmm = 0.*B.op('r=1',N,1,ell)
    else: rDmm = B.xi(-1,ell-1)*B.op('r=1',N,1,ell-2)*D(-1,0,-1)
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
    N0, N1, N2, N4, N5, N6, N7 = BC_rows(N)
    row0=np.concatenate(( QSm[1]+QSm[3], QS0[1]+QS0[3] , QSp[1]+QSp[3], np.zeros(N8-N2) ))
    row1=np.concatenate(( u0m          , np.zeros(N0+1), u0p          , np.zeros(N8-N2) ))
    row2=np.concatenate(( QSm[5]+QSm[7], QS0[5]+QS0[7] , QSp[5]+QSp[7], np.zeros(N8-N2) ))
    row3=np.concatenate(( np.zeros(N3+1), B.op('r=1',N,0,ell), np.zeros(N8-N4) ))
    row4=np.concatenate(( np.zeros(N4+1), B.op('r=1',N,0,ell-1), np.zeros(N8-N5) ))
    row5=np.concatenate(( np.zeros(N5+1), B.op('r=1',N,1,ell-1)*B.op('D-',N,0,ell), np.zeros(N8-N6) ))
    row6=np.concatenate(( np.zeros(N6+1), B.op('r=1',N,1,ell)*B.op('D-',N,0,ell+1), np.zeros(N8-N7) ))

    tau0 = (C(-1))[:,-1]
    tau1 = (C( 0))[:,-1]
    tau2 = (C( 1))[:,-1]
    tau3 = (C( 0))[:,-1]
    tau4 = (C(-1))[:,-1]
    tau5 = (C( 0))[:,-1]
    tau6 = (C( 1))[:,-1]

    tau0 = tau0.reshape((len(tau0),1))
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = tau2.reshape((len(tau2),1))
    tau3 = tau3.reshape((len(tau3),1))
    tau4 = tau4.reshape((len(tau4),1))
    tau5 = tau5.reshape((len(tau5),1))
    tau6 = tau6.reshape((len(tau6),1))

    col0 = np.concatenate((                   tau0,np.zeros((N8-N0,1))))
    col1 = np.concatenate((np.zeros((N0+1,1)),tau1,np.zeros((N8-N1,1))))
    col2 = np.concatenate((np.zeros((N1+1,1)),tau2,np.zeros((N8-N2,1))))
    col3 = np.concatenate((np.zeros((N3+1,1)),tau3,np.zeros((N8-N4,1))))
    col4 = np.concatenate((np.zeros((N4+1,1)),tau4,np.zeros((N8-N5,1))))
    col5 = np.concatenate((np.zeros((N5+1,1)),tau4,np.zeros((N8-N6,1))))
    col6 = np.concatenate((np.zeros((N6+1,1)),tau4,np.zeros((N8-N7,1))))

    L = sparse.bmat([[   L, col0, col1, col2, col3, col4, col5, col6],
                     [row0,    0,    0,    0,    0,    0,    0,    0],
                     [row1,    0,    0,    0,    0,    0,    0,    0],
                     [row2,    0,    0,    0,    0,    0,    0,    0],
                     [row3,    0,    0,    0,    0,    0,    0,    0],
                     [row4,    0,    0,    0,    0,    0,    0,    0],
                     [row5,    0,    0,    0,    0,    0,    0,    0],
                     [row6,    0,    0,    0,    0,    0,    0,    0]])

    M = sparse.bmat([[     M, 0*col0, 0*col1, 0*col2, 0*col3, 0*col4, 0*col5, 0*col6],
                     [0*row0,      0,      0,      0,      0,      0,      0,      0],
                     [0*row1,      0,      0,      0,      0,      0,      0,      0],
                     [0*row2,      0,      0,      0,      0,      0,      0,      0],
                     [0*row3,      0,      0,      0,      0,      0,      0,      0],
                     [0*row4,      0,      0,      0,      0,      0,      0,      0],
                     [0*row5,      0,      0,      0,      0,      0,      0,      0],
                     [0*row6,      0,      0,      0,      0,      0,      0,      0]])

    L = L.tocsr()
    M = M.tocsr()

    return M, L

class StateVector:

    def __init__(self,u,p,T,A,pi):
        self.data = []
        for ell in range(ell_start,ell_end+1):
            ell_local = ell-ell_start
            if ell == 0: taus = np.zeros(1)
            else: taus = np.zeros(7)
            for m in range(m_start,m_end+1):
                m_local = m-m_start
                self.data.append(np.concatenate((u['c'][ell_local][:,m_local],p['c'][ell_local][:,m_local],
                                                 T['c'][ell_local][:,m_local],A['c'][ell_local][:,m_local],
                                                pi['c'][ell_local][:,m_local],taus)))

    def pack(self,u,p,T,A,pi):
        for ell in range(ell_start,ell_end+1):
            ell_local = ell-ell_start
            if ell == 0: taus = np.zeros(1)
            else: taus = np.zeros(7)
            for m in range(m_start,m_end+1):
                m_local = m-m_start
                self.data[ell_local*m_size+m_local] = np.concatenate((u['c'][ell_local][:,m_local],
                                                                      p['c'][ell_local][:,m_local],
                                                                      T['c'][ell_local][:,m_local],
                                                                      A['c'][ell_local][:,m_local],
                                                                     pi['c'][ell_local][:,m_local],
                                                                      taus))

    def unpack(self,u,p,T,A,pi):
        for ell in range(ell_start,ell_end+1):
            ell_local = ell-ell_start
            end_u = u['c'][ell_local].shape[0]
            p_len = p['c'][ell_local].shape[0]
            T_len = T['c'][ell_local].shape[0]
            A_len = A['c'][ell_local].shape[0]
            for m in range(m_start,m_end+1):
                m_local = m - m_start
                u['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][:end_u]
                p['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][end_u:end_u+p_len]
                T['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][end_u+p_len:end_u+p_len+T_len]
                A['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][end_u+p_len+T_len:end_u+p_len+T_len+A_len]
                pi['c'][ell_local][:,m_local]= self.data[ell_local*m_size+m_local][end_u+p_len+T_len+A_len:end_u+p_len+T_len+A_len+p_len]


# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Resolution
L_max = 23
N_max = 23
R_max = 3

alpha_BC = 0

L_dealias = 3/2
N_dealias = 3/2
N_r = N_max

# parameters
Ekman = 5e-4
Rossby = 5./7.*1e-4
q = 7
Rayleigh = 200
S = 3*q

# Integration parameters
dt = 0.000005
t_end = 10

# Make domain
#mesh = [6,6]
mesh = None
phi_basis = de.Fourier('phi',2*(L_max+1), interval=(0,2*np.pi),dealias=L_dealias)
theta_basis = de.Fourier('theta', L_max+1, interval=(0,np.pi),dealias=L_dealias)
r_basis = de.Fourier('r', N_r+1, interval=(0,1),dealias=N_dealias)
domain = de.Domain([phi_basis,theta_basis,r_basis], grid_dtype=np.float64, mesh=mesh)

domain.global_coeff_shape = np.array([L_max+1,L_max+1,N_max+1])
domain.distributor = Distributor(domain,comm,mesh)

mesh = domain.distributor.mesh
if len(mesh) == 0:
    phi_layout   = domain.distributor.layouts[3]
    th_m_layout  = domain.distributor.layouts[2]
    ell_r_layout = domain.distributor.layouts[1]
    r_ell_layout = domain.distributor.layouts[1]
elif len(mesh) == 1:
    phi_layout   = domain.distributor.layouts[4]
    th_m_layout  = domain.distributor.layouts[2]
    ell_r_layout = domain.distributor.layouts[1]
    r_ell_layout = domain.distributor.layouts[1]
elif len(mesh) == 2:
    phi_layout   = domain.distributor.layouts[5]
    th_m_layout  = domain.distributor.layouts[3]
    ell_r_layout = domain.distributor.layouts[2]
    r_ell_layout = domain.distributor.layouts[1]

m_start   = th_m_layout.slices(scales=1)[0].start
m_end     = th_m_layout.slices(scales=1)[0].stop-1
m_size = m_end - m_start + 1
ell_start = r_ell_layout.slices(scales=1)[1].start
ell_end   = r_ell_layout.slices(scales=1)[1].stop-1

# set up ball
N_theta = int((L_max+1)*L_dealias)
N_r     = int((N_r+1)*N_dealias)
B = ball.Ball(N_max,L_max,N_theta=N_theta,N_r=N_r,R_max=R_max,ell_min=ell_start,ell_max=ell_end,m_min=m_start,m_max=m_end,a=0)
theta_global = B.grid(0)
r_global = B.grid(1)
z, R = r_global*np.cos(theta_global), r_global*np.sin(theta_global) # global

grid_slices = phi_layout.slices(domain.dealias)
phi = domain.grid(0,scales=domain.dealias)[grid_slices[0],:,:]
theta = B.grid(1,dimensions=3)[:,grid_slices[1],:] # local
r = B.grid(2,dimensions=3)[:,:,grid_slices[2]] # local

weight_theta = B.weight(1,dimensions=3)
weight_r = B.weight(2,dimensions=3)

Du = ball.TensorField_3D(2,B,domain)
u  = ball.TensorField_3D(1,B,domain)
p  = ball.TensorField_3D(0,B,domain)
T  = ball.TensorField_3D(0,B,domain)
DT = ball.TensorField_3D(1,B,domain)
A  = ball.TensorField_3D(1,B,domain)
H  = ball.TensorField_3D(1,B,domain)
DH = ball.TensorField_3D(2,B,domain)
pi = ball.TensorField_3D(0,B,domain)

u_rhs = ball.TensorField_3D(1,B,domain)
p_rhs = ball.TensorField_3D(0,B,domain)
T_rhs = ball.TensorField_3D(0,B,domain)
A_rhs = ball.TensorField_3D(1,B,domain)
pi_rhs= ball.TensorField_3D(0,B,domain)

# initial condition
T['g'] = 0.5*(1-r**2) + 1e-5/8.*np.sqrt(35/np.pi)*r**3*(1-r**2)*(np.cos(3*phi)+np.sin(3*phi))*np.sin(theta)**3
u['g'][1] = -10*r**2/7/np.sqrt(3)*np.cos(theta)*(  3*(-147+343*r**2-217*r**4+29*r**6)*np.cos(phi)
                                                 +14*(-9 - 125*r**2 +39*r**4+27*r**6)*np.sin(phi) )
u['g'][2] = -5*r/5544*( 7*(           (43700-58113*r**2-15345*r**4+1881*r**6+20790*r**8)*np.sin(theta)
                           +1485*r**2*(-9 + 115*r**2 - 167*r**4 + 70*r**6)*np.sin(3*theta) )
                       +528*np.sqrt(3)*r*np.cos(2*theta)*( 14*(-9-125*r**2+39*r**4+27*r**6)*np.cos(phi)
                                                           +3*(147-343*r**2+217*r**4-29*r**6)*np.sin(phi) ) )

# give initial magnetic field
H['g'][1] = -3./2.*r*(-1+4*r**2-6*r**4+3*r**6)*(np.cos(phi)+np.sin(phi))
H['g'][2] = -3./4.*r*(-1+r**2)*np.cos(theta)* \
                 ( 3*r*(2-5*r**2+4*r**4)*np.sin(theta)
                  +2*(1-3*r**2+3*r**4)*(np.cos(phi)-np.sin(phi)))

# then solve for A
for ell in range(ell_start,ell_end+1):
    ell_local = ell - ell_start
    N = B.N_max - B.N_min(ell-B.R_max)
    Dm = B.op('D-',N,0,ell+1).astype(np.complex128)
    Dp = B.op('D+',N,0,ell-1).astype(np.complex128)
    Dp[N,0] = 1. # BC
    E  = B.op('E',N,0,ell).astype(np.complex128)
    xim = B.xi(-1,ell)
    xip = B.xi(+1,ell)
    A['c'][ell_local][:(N+1)]   = spla.spsolve(Dp, 1j*xip*E.dot(H['c'][ell_local][(N+1):2*(N+1)]))
    A['c'][ell_local][2*(N+1):] = spla.spsolve(Dm,-1j*xim*E.dot(H['c'][ell_local][(N+1):2*(N+1)]))

# build state vector
state_vector = StateVector(u,p,T,A,pi)
NL = StateVector(u,p,T,A,pi)
timestepper = timesteppers.CNAB2(StateVector, u,p,T,A,pi)

# build matrices
M,L,P,LU = [],[],[],[]
for ell in range(ell_start,ell_end+1):
    N = B.N_max - B.N_min(ell-B.R_max)
    M_ell,L_ell = matrices(N,ell,Ekman,Rossby,q)
    M.append(M_ell.astype(np.complex128))
    L.append(L_ell.astype(np.complex128))
    P.append(M_ell.astype(np.complex128))
    LU.append([None])

# calculate RHS terms from state vector
def nonlinear(state_vector, RHS, t):

    # get U in coefficient space
    state_vector.unpack(u,p,T,A,pi)

    # H = curl(A)
    for ell in range(ell_start,ell_end+1):
        ell_local = ell - ell_start
        B.curl(ell,1,A['c'][ell_local],H['c'][ell_local])

    # take derivatives
    for ell in range(ell_start,ell_end+1):
        ell_local = ell - ell_start
        Du['c'][ell_local] = B.grad(ell,1,u['c'][ell_local])
        DT['c'][ell_local] = B.grad(ell,0,T['c'][ell_local])
        DH['c'][ell_local] = B.grad(ell,1,H['c'][ell_local])

    # R = ez cross u
    ez = np.array([np.cos(theta),-np.sin(theta),0*np.cos(theta)])
    u_rhs['g'] = -B.cross_grid(ez,u['g'])
    for i in range(3):
        u_rhs['g'][i] -= Rossby*(u['g'][0]*Du['g'][i] + u['g'][1]*Du['g'][3*1+i] + u['g'][2]*Du['g'][3*2+i])
        u_rhs['g'][i] +=        (H['g'][0]*DH['g'][i] + H['g'][1]*DH['g'][3*1+i] + H['g'][2]*DH['g'][3*2+i])
    u_rhs['g'][0] += q*Rayleigh*r*T['g'][0]
    p_rhs['g'] = 0.
    T_rhs['g'] = S - (u['g'][0]*DT['g'][0] + u['g'][1]*DT['g'][1] + u['g'][2]*DT['g'][2])
    A_rhs['g'] = B.cross_grid(u['g'],H['g'])
    pi_rhs['g']= 0.

    # transform (ell, r) -> (ell, N)
    for ell in range(ell_start, ell_end+1):
        ell_local = ell - ell_start

        N = N_max - B.N_min(ell-R_max)

        # multiply by conversion matrices (may be very important)
        # note that M matrices are no longer conversion matrices -- need to divide by Ekman or Prandtl
        u_len = u_rhs['c'][ell_local].shape[0]
        p_len = p_rhs['c'][ell_local].shape[0]
        T_len = T_rhs['c'][ell_local].shape[0]
        A_len = A_rhs['c'][ell_local].shape[0]
        u_rhs['c'][ell_local] = M[ell_local][:u_len,:u_len].dot(u_rhs['c'][ell_local])/Rossby
        T_rhs['c'][ell_local] = M[ell_local][u_len+p_len:u_len+p_len+T_len,u_len+p_len:u_len+p_len+T_len].dot(T_rhs['c'][ell_local])
        A_rhs['c'][ell_local] = M[ell_local][u_len+p_len+T_len:u_len+p_len+T_len+A_len,u_len+p_len+T_len:u_len+p_len+T_len+A_len].dot(A_rhs['c'][ell_local])

    NL.pack(u_rhs,p_rhs,T_rhs,A_rhs,pi_rhs)

def backward_state(state_vector):

    state_vector.unpack(u,p,T,A,pi)

    # H = curl(A)
    for ell in range(ell_start,ell_end+1):
        ell_local = ell - ell_start
        B.curl(ell,1,A['c'][ell_local],H['c'][ell_local])

    ur_global  = comm.gather(u['g'][0], root=0)
    uth_global = comm.gather(u['g'][1], root=0)
    uph_global = comm.gather(u['g'][2], root=0)
    p_global   = comm.gather(p['g'], root=0)
    T_global   = comm.gather(T['g'], root=0)
    Br_global  = comm.gather(H['g'][0], root=0)
    Bth_global = comm.gather(H['g'][1], root=0)
    Bph_global = comm.gather(H['g'][2], root=0)

    starts = comm.gather(phi_layout.start(scales=domain.dealias),root=0)
    counts = comm.gather(phi_layout.local_shape(scales=domain.dealias),root=0)

    if rank == 0:
        ur_full  = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        uth_full = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        uph_full = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        p_full   = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        T_full   = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        Br_full  = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        Bth_full = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        Bph_full = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        for i in range(size):
            spatial_slices = tuple(slice(s, s+c) for (s,c) in zip(starts[i], counts[i]))
            ur_full[spatial_slices]  = ur_global[i]
            uth_full[spatial_slices] = uth_global[i]
            uph_full[spatial_slices] = uph_global[i]
            p_full[spatial_slices]   = p_global[i]
            T_full[spatial_slices]   = T_global[i]
            Br_full[spatial_slices]  = Br_global[i]
            Bth_full[spatial_slices] = Bth_global[i]
            Bph_full[spatial_slices] = Bph_global[i]
    else:
        ur_full  = None
        uth_full = None
        uph_full = None
        p_full   = None
        T_full   = None
        Br_full  = None
        Bth_full = None
        Bph_full = None

    return ur_full,uth_full,uph_full,p_full,T_full,Br_full,Bth_full,Bph_full

t_list = []
Ek_list = []
Em_list = []

# timestepping loop
start_time = time.time()

t = 0
iter = 0

while t < t_end:

    nonlinear(state_vector,NL,t) 

    if iter % 5 == 0:
        ur_grid, uth_grid, uph_grid, p_grid, T_grid, Br_grid, Bth_grid, Bph_grid = backward_state(state_vector)
        if rank == 0:
            Ek = np.sum(weight_r*weight_theta* 0.5*(np.abs(ur_grid)**2 + np.abs(uth_grid)**2 + np.abs(uph_grid)**2) )*(np.pi)/(L_max+1)/L_dealias
            Em = np.sum(weight_r*weight_theta* 0.5/Rossby*(np.abs(Br_grid)**2 + np.abs(Bth_grid)**2 + np.abs(Bph_grid)**2) )*(np.pi)/(L_max+1)/L_dealias
            print( t,iter,Ek,Em )

        if iter % 100000 == 0 and rank == 0:
            output_num = iter // 100000
            file = open('checkpoint_L%i' %output_num, 'wb')
            for a in [ur_grid,uth_grid,uph_grid,p_grid,T_grid,Br_grid,Bth_grid,Bph_grid]:
                pickle.dump(a,file)
            file.close()

    timestepper.step(dt, state_vector, B, L, M, P, NL, LU)
    t += dt
    iter += 1

end_time = time.time()
if rank==0:
    print('simulation took: %f' %(end_time-start_time))

