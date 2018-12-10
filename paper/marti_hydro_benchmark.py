from dedalus_sphere import ball_wrapper as ball
from dedalus_sphere import ball128
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
from dedalus_sphere import timesteppers

# Gives LHS matrices for hydro.

def BC_rows(N):
    N0 = N
    N1 = N + N0 + 1
    N2 = N + N1 + 1
    return N0,N1,N2

def matrices(N,ell,Ekman):

    nu = Ekman

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

    if ell == 0: #+3 is for tau rows
        M = B.op('0',N3+3,0,ell).tocsr()
        L = B.op('I',N3+3,0,ell).tocsr()
        return M, L

    xim, xip = B.xi([-1,+1],ell)

    M00 = E(1,-1).dot(E( 0,-1))
    M11 = E(1, 0).dot(E( 0, 0))
    M22 = E(1,+1).dot(E( 0,+1))

    M=sparse.bmat([[M00, Z,   Z,  Z],
                   [Z, M11,   Z,  Z],
                   [Z,   Z, M22,  Z],
                   [Z,   Z,   Z,  Z]])
    M = M.tocsr()
    #M[N0]=np.zeros(N3+1) # for boundary conditions
    #M[N1]=np.zeros(N3+1)
    #M[N2]=np.zeros(N3+1)

    L00 = -nu*D(-1,1, 0).dot(D(+1, 0,-1))
    L11 = -nu*D(-1,1,+1).dot(D(+1, 0, 0))
    L22 = -nu*D(+1,1, 0).dot(D(-1, 0,+1))

    L03 = xim*E(+1,-1).dot(D(-1,0,0))
    L23 = xip*E(+1,+1).dot(D(+1,0,0))

    L30 = xim*D(+1,0,-1)
    L32 = xip*D(-1,0,+1)

    L=sparse.bmat([[L00,  Z,   Z, L03],
                   [Z,  L11,   Z,   Z],
                   [Z,    Z, L22, L23],
                   [L30,  Z, L32,   Z]])
    L = L.tocsr()
    N0, N1, N2 = BC_rows(N)
    row0=np.concatenate((               B.op('r=1',N,0,ell-1),np.zeros(N3-N0)))
    row1=np.concatenate((np.zeros(N0+1),B.op('r=1',N,0,ell  ),np.zeros(N3-N1)))
    row2=np.concatenate((np.zeros(N1+1),B.op('r=1',N,0,ell+1),np.zeros(N3-N2)))

    tau0 = (C(-1))[:,-1]
    tau1 = (C( 0))[:,-1]
    tau2 = (C( 1))[:,-1]

    tau0 = tau0.reshape((len(tau0),1))
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = tau2.reshape((len(tau2),1))

    col0 = np.concatenate((                   tau0,np.zeros((N3-N0,1))))
    col1 = np.concatenate((np.zeros((N0+1,1)),tau1,np.zeros((N3-N1,1))))
    col2 = np.concatenate((np.zeros((N1+1,1)),tau2,np.zeros((N3-N2,1))))

    L = sparse.bmat([[   L, col0, col1, col2],
                     [row0,    0 ,   0,    0],
                     [row1,    0 ,   0,    0],
                     [row2,    0,    0,    0]])

    M = sparse.bmat([[     M, 0*col0, 0*col1, 0*col2],
                     [0*row0,      0 ,     0,      0],
                     [0*row1,      0 ,     0,      0],
                     [0*row2,      0,      0,      0]])

    L = L.tocsr()
    M = M.tocsr()

    return M, L

class StateVector:

    def __init__(self,u,p):
        self.data = []
        taus = np.zeros(3)
        for ell in range(ell_start,ell_end+1):
            ell_local = ell-ell_start
            for m in range(m_start,m_end+1):
                m_local = m-m_start
                self.data.append(np.concatenate((u['c'][ell_local][:,m_local],p['c'][ell_local][:,m_local],taus)))

    def pack(self,u,p,BCs):
        for ell in range(ell_start,ell_end+1):
            ell_local = ell-ell_start
            for m in range(m_start,m_end+1):
                m_local = m-m_start
                BCs_m = BCs[:,m_local,ell_local]
                self.data[ell_local*m_size+m_local] = np.concatenate((u['c'][ell_local][:,m_local],
                                                                      p['c'][ell_local][:,m_local],BCs_m))

    def unpack(self,u,p):
        for ell in range(ell_start,ell_end+1):
            ell_local = ell-ell_start
            end_u = u['c'][ell_local].shape[0]
            p_len = p['c'][ell_local].shape[0]
            for m in range(m_start,m_end+1):
                m_local = m - m_start
                u['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][:end_u]
                p['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][end_u:end_u+p_len]


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
Om = 20.
u0 = np.sqrt(3/(2*np.pi))
nu = 1e-2

# Integration parameters
dt = 0.02
t_end = 40

# Make domain
mesh=[2,2]
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
B = ball.Ball(N_max,L_max,N_theta=N_theta,N_r=N_r,R_max=R_max,ell_min=ell_start,ell_max=ell_end,m_min=m_start,m_max=m_end,a=0.)
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
u = ball.TensorField_3D(1,B,domain)
p = ball.TensorField_3D(0,B,domain)

u_rhs = ball.TensorField_3D(1,B,domain)
p_rhs = ball.TensorField_3D(0,B,domain)

# create boundary conditions
BC_vec = ball.TensorField_3D(1,B,domain)
BC_vec['g'][0] = 0. # u_r = 0
BC_vec['g'][1] = - u0*r**2*np.cos(theta)*np.cos(phi)
BC_vec['g'][2] = u0*r**2*np.sin(phi)

BC_vec.require_grid_space()

local_rell_shape = r_ell_layout.local_shape(scales=domain.dealias)
BC_shape = np.append(3,np.array(local_rell_shape)[:-1])

BC_dB = np.zeros(BC_shape,dtype=np.complex128)
for ell in range(ell_start,ell_end+1):
    ell_local = ell - ell_start
    N = B.N_max - B.N_min(ell-B.R_max)
    for i in range(3):
        BC_dB[i,:,ell_local] = B.op('r=1',N,0,ell+(i-1)).dot(BC_vec['c'][ell_local][i*(N+1):(i+1)*(N+1)])

# build state vector
state_vector = StateVector(u,p)
NL = StateVector(u,p)
timestepper = timesteppers.CNAB2(StateVector, u,p)

# build matrices
M,L,P,LU = [],[],[],[]
for ell in range(ell_start,ell_end+1):
    N = B.N_max - B.N_min(ell-B.R_max)
    M_ell,L_ell = matrices(N,ell,nu)
    M.append(M_ell.astype(np.complex128))
    L.append(L_ell.astype(np.complex128))
    P.append(M_ell.astype(np.complex128))
    LU.append([None])

# calculate RHS terms from state vector
def nonlinear(state_vector, NL, t):

    # get U in coefficient space
    state_vector.unpack(u,p)

    # take derivatives
    for ell in range(ell_start,ell_end+1):
        ell_local = ell - ell_start
        Du['c'][ell_local] = B.grad(ell,1,u['c'][ell_local])

    # R = ez cross u
    ez = np.array([np.cos(theta),-np.sin(theta),0*np.cos(theta)])
    u_rhs['g'] = -Om*B.cross_grid(ez,u['g'])
    for i in range(3):
        u_rhs['g'][i] -= u['g'][0]*Du['g'][i] + u['g'][1]*Du['g'][3*1+i] + u['g'][2]*Du['g'][3*2+i]
    p_rhs['g'] = 0.

    # transform (ell, r) -> (ell, N)
    for ell in range(ell_start, ell_end+1):
        ell_local = ell - ell_start

        N = N_max - B.N_min(ell-R_max)
        N0, N1, N2 = BC_rows(N)

        M_size = u_rhs['c'][ell_local].shape[0]
        # multiply by conversion matrices (may be very important)
        u_rhs['c'][ell_local] = M[ell_local][:M_size,:M_size].dot(u_rhs['c'][ell_local])

    NL.pack(u_rhs,p_rhs,BC_dB)

def backward_state(state_vector):

    state_vector.unpack(u,p)

    ur_global  = comm.gather(u['g'][0], root=0)
    uth_global = comm.gather(u['g'][1], root=0)
    uph_global = comm.gather(u['g'][2], root=0)
    p_global   = comm.gather(p['g'][0], root=0)

    starts = comm.gather(phi_layout.start(scales=domain.dealias),root=0)
    counts = comm.gather(phi_layout.local_shape(scales=domain.dealias),root=0)

    if rank == 0:
        ur_full  = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        uth_full = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        uph_full = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        p_full   = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        for i in range(size):
            spatial_slices = tuple(slice(s, s+c) for (s,c) in zip(starts[i], counts[i]))
            ur_full[spatial_slices]  = ur_global[i]
            uth_full[spatial_slices] = uth_global[i]
            uph_full[spatial_slices] = uph_global[i]
            p_full[spatial_slices]   = p_global[i]
    else:
        ur_full  = None
        uth_full = None
        uph_full = None
        p_full   = None

    return ur_full,uth_full,uph_full,p_full

t_list = []
E_list = []

# timestepping loop
t = 0.
start_time = time.time()
iter = 0

while t < t_end:

    nonlinear(state_vector, NL, t)

    if iter % 10 == 0:
        ur_grid, uth_grid, uph_grid, p_grid = backward_state(state_vector)
        if rank == 0:
            E0 = np.sum(weight_r*weight_theta*(ur_grid**2+uth_grid**2+uph_grid**2) )
            E0 = 0.5*E0*(np.pi)/(L_max+1)/L_dealias

            print(t,E0)
            t_list.append(t)
            E_list.append(E0)

    timestepper.step(dt, state_vector, B, L, M, P, NL, LU)

    t += dt
    iter += 1

end_time = time.time()

if rank==0:
    print('simulation took: %f' %(end_time-start_time))
    t_list = np.array(t_list)
    E_list = np.array(E_list)
    np.savetxt('marti_E.dat',np.array([t_list,E_list]))
