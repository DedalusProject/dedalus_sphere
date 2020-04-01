import numpy as np
import scipy.sparse      as sparse
import dedalus.public as de
from dedalus.extras.flow_tools import GlobalArrayReducer
from mpi4py import MPI
import time
from dedalus_sphere import timesteppers, ball, intertwiner
from dedalus_sphere import jacobi128 as jacobi

import logging
logger = logging.getLogger(__name__)

rank = MPI.COMM_WORLD.rank

# Gives LHS matrices for boussinesq

def BC_rows(N,ell,num_comp):
    N_list = (np.arange(num_comp)+1)*(N - ell//2 + 1)
    return N_list

class Subproblem:

    def __init__(self,ell):
        self.ell = ell

def matrices(N, l, Ekman, Prandtl, Rayleigh):

    def C(deg):
        ab = (alpha_BC,l+deg+0.5)
        cd = (2,       l+deg+0.5)
        return jacobi.coefficient_connection(N - l//2,ab,cd)

    sp = Subproblem(l)

    Z00 = de.operators.ZeroMatrix(u, c, (c,)).subproblem_matrix(sp)
    Z01 = de.operators.ZeroMatrix(p, c, (c,)).subproblem_matrix(sp)
    Z02 = de.operators.ZeroMatrix(T, c, (c,)).subproblem_matrix(sp)
    Z10 = de.operators.ZeroMatrix(u, c, ()).subproblem_matrix(sp)
    Z11 = de.operators.ZeroMatrix(p, c, ()).subproblem_matrix(sp)
    Z12 = de.operators.ZeroMatrix(T, c, ()).subproblem_matrix(sp)
    Z20 = de.operators.ZeroMatrix(u, c, ()).subproblem_matrix(sp)
    Z21 = de.operators.ZeroMatrix(p, c, ()).subproblem_matrix(sp)

    if l == 0:
        N0, N1, N2 = BC_rows(N,l,3)

        M22 = Prandtl*de.operators.convert(T, (bk2,)).subproblem_matrix(sp)

        M = sparse.bmat([[Z00, Z01, Z02],
                         [Z10, Z11, Z12],
                         [Z20, Z21, M22]]).tocsr()

        L00 = np.eye(Z00.shape[0])
        L11 = np.eye(Z11.shape[0])
        L22 = -de.operators.Laplacian(T, c).subproblem_matrix(sp)

        L = sparse.bmat([[L00,Z01,Z02],
                         [Z10,L11,Z12],
                         [Z20,Z21,L22]]).tocsr()

        op = de.operators.interpolate(T,r=1)
        R = op.subproblem_matrix(sp)
        Z0 = de.operators.ZeroVector(u, c, ()).subproblem_matrix(sp)
        Z1 = de.operators.ZeroVector(p, c, ()).subproblem_matrix(sp)
        B_rows=np.bmat([[Z0, Z1, R]])

        tau0 = C(0)[:,-1]
        tau0 = tau0.reshape((len(tau0),1))

        col0 = np.concatenate((np.zeros((N1,1)),tau0))

        L = sparse.bmat([[     L, col0],
                         [B_rows,    0]])

        M = sparse.bmat([[       M, 0*col0],
                         [0*B_rows,      0]])

        L = L.tocsr()
        M = M.tocsr()

        return M, L

    M00 =   Ekman*de.operators.convert(u, (bk2,)).subproblem_matrix(sp)
    M22 = Prandtl*de.operators.convert(T, (bk2,)).subproblem_matrix(sp)

    M = sparse.bmat([[M00, Z01, Z02],
                     [Z10, Z11, Z12],
                     [Z20, Z21, M22]]).tocsr()

    Lu = de.operators.convert(-Ekman*de.operators.Laplacian(u, c) + de.operators.Gradient(p, c), (bk2,))
    op_matrices = Lu.expression_matrices(sp, (u,p,))
    L00 = op_matrices[u]
    L01 = op_matrices[p]
    L10 = de.operators.Divergence(u).subproblem_matrix(sp)
    L22 = -de.operators.Laplacian(T, c).subproblem_matrix(sp)

    L = sparse.bmat([[L00,L01,Z02],
                     [L10,Z11,Z12],
                     [Z20,Z21,L22]]).tocsr()

    N0, N1, N2, N3, N4 = BC_rows(N,l,5)

    Q = b.radial_recombinations((c,c,),ell_list=(l,))[0]
    if l == 1: rDmm = 0.*b.operator_matrix('r=R',l,0)
    else: rDmm = intertwiner.xi(-1,l-1)*b.operator_matrix('r=R',l,-2,dk=1) @ b.operator_matrix('D-',l,-1)
    rDpm = intertwiner.xi(+1,l-1)*b.operator_matrix('r=R',l, 0,dk=1) @ b.operator_matrix('D+',l,-1)
    rDm0 = intertwiner.xi(-1,l  )*b.operator_matrix('r=R',l,-1,dk=1) @ b.operator_matrix('D-',l, 0)
    rDp0 = intertwiner.xi(+1,l  )*b.operator_matrix('r=R',l,+1,dk=1) @ b.operator_matrix('D+',l, 0)
    rDmp = intertwiner.xi(-1,l+1)*b.operator_matrix('r=R',l, 0,dk=1) @ b.operator_matrix('D-',l,+1)
    rDpp = intertwiner.xi(+1,l+1)*b.operator_matrix('r=R',l,+2,dk=1) @ b.operator_matrix('D+',l,+1)

    rD = np.array([rDmm, rDmp, rDm0, rDpm, rDpp, rDp0, 0.*rDmm, 0.*rDmm, 0.*rDmm])
    QSm = Q[:,::3].dot(rD[::3])
    QSp = Q[:,1::3].dot(rD[1::3])
    QS0 = Q[:,2::3].dot(rD[2::3])
    Q = b.radial_recombinations((c,),ell_list=(l,))[0]
    u0m = Q[2,0]*b.operator_matrix('r=R',l,-1)
    u0p = Q[2,1]*b.operator_matrix('r=R',l,+1)

    row0=np.concatenate(( QSm[2]+QSm[6], QSp[2]+QSp[6], QS0[2]+QS0[6], np.zeros(N4-N2)))
    row1=np.concatenate(( u0m          , u0p          , np.zeros(N4-N1)))
    row2=np.concatenate(( QSm[5]+QSm[7], QSp[5]+QSp[7], QS0[5]+QS0[7], np.zeros(N4-N2)))
    row3=np.concatenate(( np.zeros(N3), b.operator_matrix('r=R',l,0) ))

    tau0 = C(-1)[:,-1]
    tau1 = C( 0)[:,-1]
    tau2 = C( 1)[:,-1]
    tau3 = C( 0)[:,-1]

    tau0 = tau0.reshape((len(tau0),1))
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = tau2.reshape((len(tau2),1))
    tau3 = tau3.reshape((len(tau3),1))

    col0 = np.concatenate((                   tau0,np.zeros((N4-N0,1))))
    col1 = np.concatenate((np.zeros((N0,1)),tau1,np.zeros((N4-N1,1))))
    col2 = np.concatenate((np.zeros((N1,1)),tau2,np.zeros((N4-N2,1))))
    col3 = np.concatenate((np.zeros((N3,1)),tau3))

    L = sparse.bmat([[   L, col0, col1, col2, col3],
                     [row0,    0 ,   0,    0,    0],
                     [row1,    0 ,   0,    0,    0],
                     [row2,    0,    0,    0,    0],
                     [row3,    0,    0,    0,    0]])

    M = sparse.bmat([[     M, 0*col0, 0*col1, 0*col2, 0*col3],
                     [0*row0,      0,      0,      0,      0],
                     [0*row1,      0,      0,      0,      0],
                     [0*row2,      0,      0,      0,      0],
                     [0*row3,      0,      0,      0,      0]])

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
            if l == 0:
                BCs = 1
            else:
                BCs = 4

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

    def pack(self,fields):
        for dl,l in enumerate(self.basis.local_l):
            for dm,m in enumerate(self.basis.local_m):
                if m <= l:
                    for i, component in enumerate(self.component_list):
                        field_num, field_component = component[0], component[1]
                        reg = field_component
                        n_slice = self.basis.n_slice(field_component,l)
                        if n_slice is not None:
                            self.data[dl][dm][self.slices[dl][i]] = fields[field_num]['c'][field_component][dm,dl,n_slice]

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


Lmax = 15
Nmax = 15

# right now can't run with dealiasing
L_dealias = 1
N_dealias = 1

alpha_BC = 0

# parameters
Ekman = 3e-4
Prandtl = 1
Rayleigh = 95
S = 3

# Integration parameters
dt = 8e-5
t_end = 20

c = de.coords.SphericalCoordinates('phi', 'theta', 'r')
d = de.distributor.Distributor(c.coords)
b    = de.basis.BallBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radius=1)
bk1  = de.basis.BallBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), k=1, radius=1)
bk2  = de.basis.BallBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), k=2, radius=1)
phi, theta, r = b.local_grids((1, 1, 1))

weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radius_weights(1)

u = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
p = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)
T = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)

ez = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

r_vec = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
r_vec['g'][2] = r

Source = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)
Source['g'] = S

u_rhs = de.field.Field(dist=d, bases=(bk2,), tensorsig=(c,), dtype=np.complex128)
p_rhs = de.field.Field(dist=d, bases=(bk2,), dtype=np.complex128)
T_rhs = de.field.Field(dist=d, bases=(bk2,), dtype=np.complex128)

# initial condition
T['g'] = 0.5*(1-r**2) + 0.1/8.*np.sqrt(35/np.pi)*r**3*(1-r**2)*(np.cos(3*phi)+np.sin(3*phi))*np.sin(theta)**3

# build state vector
state_vector = StateVector((u,p,T))
NL = StateVector((u,p,T))
timestepper = timesteppers.SBDF4(StateVector, (u,p,T))
state_vector.pack((u,p,T))

# build matrices
M,L,P,LU = [],[],[],[]
for l in b.local_l:
    M_ell,L_ell = matrices(Nmax,l,Ekman,Prandtl,Rayleigh)
    M.append(M_ell.astype(np.complex128))
    L.append(L_ell.astype(np.complex128))
    P.append(M_ell.astype(np.complex128))
    LU.append([None])

op_u = -de.operators.CrossProduct(ez,u) - Ekman*de.operators.DotProduct(u,de.operators.Gradient(u, c)) + Rayleigh*r_vec*T
u_rhs_op = de.operators.convert(op_u,(bk2,))
op_T = Source - Prandtl*de.operators.DotProduct(u,de.operators.Gradient(T, c))
T_rhs_op = de.operators.convert(op_T,(bk2,))

# calculate RHS terms from state vector
def nonlinear(state_vector, NL, t):

    # get U in coefficient space
    state_vector.unpack((u,p,T))

    u_rhs = u_rhs_op.evaluate()
    if 0 in b.local_l:
        u_rhs['c'][:,:,0,:] = 0 # very important to zero out the ell=0 RHS
    T_rhs = T_rhs_op.evaluate()

    NL.pack((u_rhs,p_rhs,T_rhs))

reducer = GlobalArrayReducer(d.comm_cart)

vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

t = 0.

t_list = []
E_list = []

# timestepping loop
start_time = time.time()
iter = 0

while t < t_end:

    nonlinear(state_vector,NL,t)

    if iter % 10 == 0:
        E0 = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        E0 = 0.5*E0*(np.pi)/(Lmax+1)/L_dealias
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        logger.info("iter: {:d}, dt={:e}, t={:e}, E0={:e}".format(iter, dt, t,E0))
        t_list.append(t)
        E_list.append(E0)

    timestepper.step(dt, state_vector, b, L, M, P, NL, LU)
    t += dt
    iter += 1

end_time = time.time()
if rank==0:
    print('simulation took: %f' %(end_time-start_time))
    t_list = np.array(t_list)
    E_list = np.array(E_list)
    np.savetxt('marti_conv.dat',np.array([t_list,E_list]))

