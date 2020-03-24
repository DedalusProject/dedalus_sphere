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

class Subproblem:

    def __init__(self,ell):
        self.ell = ell

def matrices(N,l,nu):

    sp = Subproblem(l)

    def C(deg):
        ab = (alpha_BC,l+deg+0.5)
        cd = (2,       l+deg+0.5)
        return jacobi.connection(N - max((l + deg)//2,0),ab,cd)

    N0, N1, N2, N3 = BC_rows(N,l,[-1,+1,0,0])

    if l == 0: #+3 is for tau rows
        N0, N1 = BC_rows(N,l,[+1,0])
        M = ball.operator(3,'0',N1-1+3,0,l,0).tocsr()
        L = ball.operator(3,'I',N1-1+3,0,l,0).tocsr()
        return M, L

    op = de.operators.convert(u, (bk2,))
    op_matrices = op.expression_matrices(sp, (u,))
    M00 = op_matrices[u]

    M01 = de.operators.Zero(p, c, (c,)).subproblem_matrix(sp)
    M10 = de.operators.Zero(u, c, ()).subproblem_matrix(sp)
    M11 = de.operators.Zero(p, c, ()).subproblem_matrix(sp)

    M=sparse.bmat([[M00,M01],
                   [M10,M11]])

    M.tocsr()

    op = de.operators.convert(-nu*de.operators.Laplacian(u, c) + de.operators.Gradient(p, c), (bk2,))
    op_matrices = op.expression_matrices(sp, (u,p,))
    L00 = op_matrices[u]
    L01 = op_matrices[p]

    op = de.operators.Divergence(u)
    L10 = op.subproblem_matrix(sp)

    L11 = de.operators.Zero(p, c, ()).subproblem_matrix(sp)

    L=sparse.bmat([[L00,L01],
                   [L10,L11]])

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

ez = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta) 

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
                BCs[i,dm,dl] = ball.operator(3,'r=R',Nmax,0,l,b.regtotal((i,))).astype(np.float64) @ u_bc['c'][i,dm,dl,n_slice]

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

op = -Om*de.operators.CrossProduct(ez,u) - de.operators.DotProduct(u,de.operators.Gradient(u, c))
conv_op = de.operators.convert(op,(bk2,))

# calculate RHS terms from state vector
def nonlinear(state_vector, NL, t):

    # get U in coefficient space
    state_vector.unpack((u,p))
    u_rhs = conv_op.evaluate()
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

    nonlinear(state_vector, NL, t)

    if iter % 10 == 0:
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

