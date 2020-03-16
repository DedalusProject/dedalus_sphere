from dedalus_sphere import ball
from dedalus_sphere import timesteppers
from dedalus_sphere import intertwiner
from dedalus_sphere import jacobi128 as jacobi
import numpy as np
import scipy.sparse      as sparse
from dedalus.extras.flow_tools import GlobalArrayReducer
import dedalus.public as de
from mpi4py import MPI
import time
import pickle

import logging
logger = logging.getLogger(__name__)

rank = MPI.COMM_WORLD.rank

# Gives LHS matrices for hydro.

def BC_rows(N,ell,deg):
    N_list = []
    for d in deg:
        N_list.append( N - max((ell + d)//2,0) + 1 )
    if len(deg) == 1: return N_list
    N_list = np.cumsum(N_list)
    return N_list

def matrices(N,l,nu):

    def D(mu,i,deg):
        if mu == +1: return ball.operator(3,'D+',N,i,l,deg)
        if mu == -1: return ball.operator(3,'D-',N,i,l,deg)

    def E(i,deg): return ball.operator(3,'E',N,i,l,deg)

    def Z(deg_out,deg_in): return ball.zeros(N,l,deg_out,deg_in)

    def C(deg):
        ab = (alpha_BC,l+deg+0.5)
        cd = (2,       l+deg+0.5)
        return jacobi.connection(N - max((l + deg)//2,0),ab,cd)

    Z00 = Z(-1,-1)
    Z01 = Z(-1,+1)
    Z02 = Z(-1, 0)
    Z03 = Z(-1, 0)
    Z10 = Z(+1,-1)
    Z11 = Z(+1,+1)
    Z12 = Z(+1, 0)
    Z13 = Z(+1, 0)
    Z20 = Z( 0,-1)
    Z21 = Z( 0,+1)
    Z22 = Z( 0, 0)
    Z23 = Z( 0, 0)
    Z30 = Z( 0,-1)
    Z31 = Z( 0,+1)
    Z32 = Z( 0, 0)
    Z33 = Z( 0, 0)

    N0, N1, N2, N3 = BC_rows(N,l,[-1,+1,0,0])

    if l == 0: #+3 is for tau rows
        N0, N1 = BC_rows(N,l,[+1,0])
        M = ball.operator(3,'0',N1-1+3,0,l,0).tocsr()
        L = ball.operator(3,'I',N1-1+3,0,l,0).tocsr()
        return M, L

    xim, xip = intertwiner.xi([-1,+1],l)

    M00 = E(1,-1).dot(E( 0,-1))
    M11 = E(1,+1).dot(E( 0,+1))
    M22 = E(1, 0).dot(E( 0, 0))

    M=sparse.bmat([[M00, Z01, Z02, Z03],
                   [Z10, M11, Z12, Z13],
                   [Z20, Z21, M22, Z23],
                   [Z30, Z31, Z32, Z33]])
    M.tocsr()

    L00 = -nu*D(-1,1, 0).dot(D(+1, 0,-1))
    L11 = -nu*D(+1,1, 0).dot(D(-1, 0,+1))
    L22 = -nu*D(-1,1,+1).dot(D(+1, 0, 0))

    L03 = xim*E(+1,-1).dot(D(-1,0,0))
    L13 = xip*E(+1,+1).dot(D(+1,0,0))

    L30 = xim*D(+1,0,-1)
    L31 = xip*D(-1,0,+1)

    L=sparse.bmat([[L00, Z01, Z02, L03],
                   [Z10, L11, Z12, L13],
                   [Z20, Z21, L22, Z23],
                   [L30, L31, Z32, Z33]])
    L = L.tocsr()

    row0=np.concatenate((             ball.operator(3,'r=R',N,0,l,-1),np.zeros(N3-N0)))
    row1=np.concatenate((np.zeros(N0),ball.operator(3,'r=R',N,0,l,+1),np.zeros(N3-N1)))
    row2=np.concatenate((np.zeros(N1),ball.operator(3,'r=R',N,0,l, 0),np.zeros(N3-N2)))

    tau0 = (C(-1))[:,-1]
    tau1 = (C(+1))[:,-1]
    tau2 = (C( 0))[:,-1]

    tau0 = tau0.reshape((len(tau0),1))
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = tau2.reshape((len(tau2),1))

    col0 = np.concatenate((                 tau0,np.zeros((N3-N0,1))))
    col1 = np.concatenate((np.zeros((N0,1)),tau1,np.zeros((N3-N1,1))))
    col2 = np.concatenate((np.zeros((N1,1)),tau2,np.zeros((N3-N2,1))))

    if l % 2 == 1:
        col0[-1] = 1.

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

    def __init__(self,fields):
        # get a list of fields
        # BCs is a function which returns the number of BCs for a given l
        self.basis = fields[0].bases[0]
        self.Nmax = self.basis.Nmax
        self.component_list = self.components(fields)
        self.data = []
        self.slices = []
        for dl,l in enumerate(self.basis.local_l):
            BCs = 3

            pencil_length = 0
            slices_l = []
            for component in self.component_list:
                field_num, field_component = component[0], component[1]
                if self.basis.regularity_allowed(l,field_component):
                    component_size = self.basis.n_size(field_component,l)
                    slices_l.append(slice(pencil_length, pencil_length+component_size))
                    pencil_length += component_size
                else:
                    slices_l.append(())
            self.slices.append(slices_l)

            l_pencil = []
            for dm,m in enumerate(self.basis.local_m):
                if m > l: l_pencil.append(None)
                else:
                    l_pencil.append(np.zeros(pencil_length+BCs, dtype=np.complex128))
            self.data.append(l_pencil)

    def components(self,fields):
        components = []
        for i, field in enumerate(fields):
            if len(field.tensorsig) == 0:
                components.append([i,()])
            elif len(field.tensorsig) == 1:
                components.append([i,(0,)])
                components.append([i,(1,)])
                components.append([i,(2,)])
        return components

    def pack(self,fields,BCs):
        for dl,l in enumerate(self.basis.local_l):
            for dm,m in enumerate(self.basis.local_m):
                if m <= l:
                    for i, component in enumerate(self.component_list):
                        field_num, field_component = component[0], component[1]
                        reg = field_component
                        n_slice = self.basis.n_slice(field_component,l)
                        if n_slice is not None:
                            self.data[dl][dm][self.slices[dl][i]] = fields[field_num]['c'][field_component][dm,dl,n_slice]
                    BC_len = BCs.shape[0]
                    self.data[dl][dm][-BC_len:] = BCs[:,dm,dl]

    def unpack(self,fields):
        for field in fields:
            field.set_layout(field.dist.coeff_layout)
        for dl,l in enumerate(self.basis.local_l):
            for dm,m in enumerate(self.basis.local_m):
                if m <= l:
                    for i, component in enumerate(self.component_list):
                        field_num, field_component = component[0], component[1]
                        n_slice = self.basis.n_slice(field_component,l)
                        if n_slice is not None:
                            fields[field_num]['c'][field_component][dm,dl,n_slice] = self.data[dl][dm][self.slices[dl][i]]


# Resolution
Lmax = 15
Nmax = 15

alpha_BC = 0

# need to figure out how to do this
L_dealias = 1
N_dealias = 1
N_r = Nmax

# parameters
Om = 20.
u0 = np.sqrt(3/(2*np.pi))
nu = 1e-2

# Integration parameters
dt = 0.02
t_end = 20

c = de.coords.SphericalCoordinates('phi', 'theta', 'r')
d = de.distributor.Distributor(c.coords)
b   = de.basis.BallBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radius=1)
bk1 = de.basis.BallBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), k=1, radius=1)
bk2 = de.basis.BallBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), k=2, radius=1)
phi, theta, r = b.local_grids((1, 1, 1))

u = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
Du = de.field.Field(dist=d, bases=(bk1,), tensorsig=(c,c,), dtype=np.complex128)
p = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)

u_rhs = de.field.Field(dist=d, bases=(bk2,), tensorsig=(c,), dtype=np.complex128)
p_rhs = de.field.Field(dist=d, bases=(bk2,), dtype=np.complex128)

weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radius_weights(1)

# create boundary conditions
u_bc = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
u_bc['g'][2] = 0. # u_r = 0
u_bc['g'][1] = - u0*r**2*np.cos(theta)*np.cos(phi)
u_bc['g'][0] = u0*r**2*np.sin(phi)

BC_shape = u_bc['c'][:,:,:,0].shape

BCs = np.zeros(BC_shape, dtype=np.complex128)

for dm, m in enumerate(b.local_m):
    for dl, l in enumerate(b.local_l):
        for i in range(3):
            if l > 0:
                n_slice = b.n_slice((i,),l)
                BCs[i,dm,dl] = ball.operator(3,'r=R',Nmax,0,l,b.regtotal(i)).astype(np.float64) @ u_bc['c'][i,dm,dl,n_slice]

# build state vector
state_vector = StateVector((u,p))
NL = StateVector((u,p))
timestepper = timesteppers.CNAB2(StateVector, (u,p))

# build matrices
M,L,P,LU = [],[],[],[]
for l in b.local_l:
    M_ell,L_ell = matrices(Nmax,l,nu)
    M.append(M_ell.astype(np.complex128))
    L.append(L_ell.astype(np.complex128))
    P.append(M_ell.astype(np.complex128))
    LU.append([None])

def cross_grid(a,b): #left handed!!!!
    return np.array([-a[1]*b[2]+a[2]*b[1],-a[2]*b[0]+a[0]*b[2],-a[0]*b[1]+a[1]*b[0]])

# calculate RHS terms from state vector
def nonlinear(state_vector, NL, t):

    # get U in coefficient space
    state_vector.unpack((u,p))

    Du.set_layout(Du.dist.coeff_layout)
    Du['c'] = 0
    op = de.operators.Gradient(u, c)
    op.out = Du
    op.evaluate()

    # R = ez cross u
    ez = np.array([0*np.cos(theta),-np.sin(theta),np.cos(theta)])
    u_rhs.set_layout(u_rhs.dist.grid_layout)
    u_rhs['g'] = -Om*cross_grid(ez,u['g'])

    for i in range(3):
        u_rhs['g'] -= u['g'][i]*Du['g'][i,:]

    u_rhs['c'][:,:,0,:] = 0 # very important to zero out the ell=0 RHS

    NL.pack((u_rhs,p_rhs),BCs)

t_list = []
E_list = []

reducer = GlobalArrayReducer(d.comm_cart)

# timestepping loop
t = 0.
start_time = time.time()
iter = 0

vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

while t < t_end:
#while iter < 11:

    nonlinear(state_vector, NL, t)

    if iter % 1 == 0:
        E0 = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        E0 = 0.5*E0*(np.pi)/(Lmax+1)/L_dealias
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        logger.info("t = %f, E = %e" %(t,E0))
        t_list.append(t)
        E_list.append(E0)

    timestepper.step(dt, state_vector, b, L, M, P, NL, LU)

    t += dt
    iter += 1

end_time = time.time()

logger.info("simulation took: %f" %(end_time-start_time))

if rank==0:
    t_list = np.array(t_list)
    E_list = np.array(E_list)
    np.savetxt('marti_E_new.dat',np.array([t_list,E_list]))

