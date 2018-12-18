"""
Dedalus script for full sphere boussinesq convection,
based on Marti convective benchmark.

Usage:
    marti_conv_benchmark.py [options]

Options:
    --Ekman=<Ekman>                      Ekman number    [default: 3e-4]
    --Rayleigh=<Rayleigh>                Rayleigh number [default: 95]
    --Prandtl=<Prandtl>                  Prandtl number  [default: 1]
    --L_max=<L_max>                      Max spherical harmonic [default: 31]
    --N_max=<N_max>                      Max radial polynomial  [default: 31]
    --mesh=<mesh>                        Processor mesh for 3-D runs

    --run_time_diffusion=<run_time_d>    How long to run, in diffusion times [default: 20]
    --run_time_iter=<run_time_i>         How long to run, in iterations

    --label=<label>                      Additional label for run output directory
"""

import ball_wrapper as ball
import ball128
import numpy as np
from   scipy.linalg      import eig
from scipy.sparse        import linalg as spla
import scipy.sparse      as sparse
import scipy.special     as spec
import dedalus.public as de
from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.core.distributor import Distributor
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import timesteppers as timesteppers

import logging
logger = logging.getLogger(__name__)

accelerate_layout = True

# Gives LHS matrices for boussinesq
def BC_rows(N):
    N0 = N
    N1 = N + N0 + 1
    N2 = N + N1 + 1
    N3 = N + N2 + 1
    N4 = N + N3 + 1
    return N0,N1,N2,N4

def matrices(N,ell,Ekman,Prandtl,Rayleigh):

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

    if ell == 0:
        I = B.op('I',N,0,ell).tocsr()
        M44 = Prandtl*E(1, 0).dot(E( 0, 0))
        M = sparse.bmat([[Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,M44]]).tocsr()
        L = sparse.bmat([[I,Z,Z,Z,Z],
                         [Z,I,Z,Z,Z],
                         [Z,Z,I,Z,Z],
                         [Z,Z,Z,I,Z],
                         [Z,Z,Z,Z,-D(-1,1,+1).dot(D(+1, 0, 0))]]).tocsr()

        row0=np.concatenate(( np.zeros(N3+1), B.op('r=1',N,0,ell) ))

        tau0 = C(0)[:,-1]
        tau0 = tau0.reshape((len(tau0),1))

        col0 = np.concatenate((np.zeros((N3+1,1)),tau0))

        L = sparse.bmat([[   L, col0],
                         [row0,    0]])

        M = sparse.bmat([[     M, 0*col0],
                         [0*row0,      0]])

        L = L.tocsr()
        M = M.tocsr()

        return M, L

    xim, xip = B.xi([-1,+1],ell)

    M00 = Ekman*E(1,-1).dot(E( 0,-1))
    M11 = Ekman*E(1, 0).dot(E( 0, 0))
    M22 = Ekman*E(1,+1).dot(E( 0,+1))
    M44 = Prandtl*E(1, 0).dot(E( 0, 0))

    M=sparse.bmat([[M00, Z,   Z,  Z,   Z],
                   [Z, M11,   Z,  Z,   Z],
                   [Z,   Z, M22,  Z,   Z],
                   [Z,   Z,   Z,  Z,   Z],
                   [Z,   Z,   Z,  Z, M44]])
    M = M.tocsr()

    L00 = -Ekman*D(-1,1, 0).dot(D(+1, 0,-1))
    L11 = -Ekman*D(-1,1,+1).dot(D(+1, 0, 0))
    L22 = -Ekman*D(+1,1, 0).dot(D(-1, 0,+1))
    L44 = -D(-1,1,+1).dot(D(+1, 0, 0))

    L03 = xim*E(+1,-1).dot(D(-1,0,0))
    L23 = xip*E(+1,+1).dot(D(+1,0,0))

    L30 = xim*D(+1,0,-1)
    L32 = xip*D(-1,0,+1)

    L04 = Z
    L24 = Z

    L=sparse.bmat([[L00,  Z,   Z, L03, L04],
                   [Z,  L11,   Z,   Z,   Z],
                   [Z,    Z, L22, L23, L24],
                   [L30,  Z, L32,   Z,   Z],
                   [Z,    Z,   Z,   Z, L44]])
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
    N0, N1, N2, N4 = BC_rows(N)

    row0=np.concatenate(( QSm[1]+QSm[3], QS0[1]+QS0[3] , QSp[1]+QSp[3], np.zeros(N4-N2)))
    row1=np.concatenate(( u0m          , np.zeros(N0+1), u0p          , np.zeros(N4-N2)))
    row2=np.concatenate(( QSm[5]+QSm[7], QS0[5]+QS0[7] , QSp[5]+QSp[7], np.zeros(N4-N2)))
    row3=np.concatenate(( np.zeros(N3+1), B.op('r=1',N,0,ell) ))

    tau0 = C(-1)[:,-1]
    tau1 = C( 0)[:,-1]
    tau2 = C( 1)[:,-1]
    tau3 = C( 0)[:,-1]

    tau0 = tau0.reshape((len(tau0),1))
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = tau2.reshape((len(tau2),1))
    tau3 = tau3.reshape((len(tau3),1))

    col0 = np.concatenate((                   tau0,np.zeros((N4-N0,1))))
    col1 = np.concatenate((np.zeros((N0+1,1)),tau1,np.zeros((N4-N1,1))))
    col2 = np.concatenate((np.zeros((N1+1,1)),tau2,np.zeros((N4-N2,1))))
    col3 = np.concatenate((np.zeros((N3+1,1)),tau3))

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

    def __init__(self,u,p,T):
        self.data = []
        for ell in range(ell_start,ell_end+1):
            if ell == 0: taus = np.zeros(1)
            else: taus = np.zeros(4)
            ell_local = ell-ell_start
            for m in range(m_start,m_end+1):
                m_local = m-m_start
                self.data.append(np.concatenate((u['c'][ell_local][:,m_local],p['c'][ell_local][:,m_local],
                                                 T['c'][ell_local][:,m_local],taus)))

    def pack(self,u,p,T):
        for ell in range(ell_start,ell_end+1):
            if ell == 0: taus = np.zeros(1)
            else: taus = np.zeros(4)
            ell_local = ell-ell_start
            for m in range(m_start,m_end+1):
                m_local = m-m_start
                self.data[ell_local*m_size+m_local] = np.concatenate((u['c'][ell_local][:,m_local],
                                                                      p['c'][ell_local][:,m_local],
                                                                      T['c'][ell_local][:,m_local],
                                                                      taus))

    def unpack(self,u,p,T):
        if accelerate_layout:
            u.layout = 'c'
            p.layout = 'c'
            T.layout = 'c'

        for ell in range(ell_start,ell_end+1):
            ell_local = ell-ell_start
            end_u = u['c'][ell_local].shape[0]
            p_len = p['c'][ell_local].shape[0]
            for m in range(m_start,m_end+1):
                m_local = m - m_start
                u['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][:end_u]
                p['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][end_u:end_u+p_len]
                T['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][end_u+p_len:end_u+2*p_len]


# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

import sys
import os
from docopt import docopt
args = docopt(__doc__)

first_time = time.time()

# Resolution
L_max = int(args['--L_max'])
N_max = int(args['--N_max'])
R_max = 3

alpha_BC = 0

L_dealias = 3/2
N_dealias = 3/2
N_r = N_max

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(size)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))

# parameters
Rayleigh = float(args['--Rayleigh'])
Ekman = float(args['--Ekman'])
Prandtl = float(args['--Prandtl'])
S = 3

data_dir = sys.argv[0].split('.py')[0]
data_dir += '_Ra{}_Ek{}_Pr{}'.format(args['--Rayleigh'],args['--Ekman'],args['--Prandtl'])
if args['--label'] == None:
    data_dir += '/'
else:
    data_dir += '_{}/'.format(args['--label'])

logger.info(sys.argv)
logger.info('-'*40)
logger.info("Run parameters")
for key in args:
    logger.info("{} = {}".format(key, args[key]))
logger.info('-'*40)
logger.info("Ra = {}, Ek = {}, Pr = {}".format(Rayleigh, Ekman, Prandtl))

if rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))

# Integration parameters
dt = 1e-5*(95/Rayleigh) #min(1e-5, Ekman/10)
t_end = float(args['--run_time_diffusion'])

# Make domain
phi_basis = de.Fourier('phi',2*(L_max+1), interval=(0,2*np.pi),dealias=L_dealias)
theta_basis = de.Fourier('theta', L_max+1, interval=(0,np.pi),dealias=L_dealias)
r_basis = de.Fourier('r', N_max+1, interval=(0,1),dealias=N_dealias)
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

# coordinate arrays for plotting
theta_global = B.grid(0)
r_global = B.grid(1)
n_phi_global = 2*(L_max+1)*L_dealias
phi_global = np.expand_dims(np.linspace(0, 2*np.pi, num=n_phi_global+1, endpoint=True), axis=1)
r_global = np.pad(r_global, ((0,0),(1,1)), mode='constant', constant_values=(0,1))
theta_global = np.pad(theta_global, ((1,1), (0,0)), mode='constant', constant_values=(np.pi,0))
logger.debug(' r: {}\n{}'.format(r_global.shape, r_global[0,:]))
logger.debug('th: {}\n{}'.format(theta_global.shape, theta_global[0,:]))
logger.debug('ph: {}\n{}'.format(phi_global.shape, phi_global[0,:]))

z, R = r_global*np.cos(theta_global), r_global*np.sin(theta_global) # global

grid_slices = phi_layout.slices(domain.dealias)
phi = domain.grid(0,scales=domain.dealias)[grid_slices[0],:,:]
theta = B.grid(1,dimensions=3)[:,grid_slices[1],:] # local
r = B.grid(2,dimensions=3)[:,:,grid_slices[2]] # local

weight_theta = B.weight(1,dimensions=3)[:,grid_slices[1],:]
weight_r = B.weight(2,dimensions=3)[:,:,grid_slices[2]]

om = ball.TensorField_3D(1,B,domain)
u  = ball.TensorField_3D(1,B,domain)
p  = ball.TensorField_3D(0,B,domain)
T  = ball.TensorField_3D(0,B,domain)
DT = ball.TensorField_3D(1,B,domain)

u_rhs = ball.TensorField_3D(1,B,domain)
p_rhs = ball.TensorField_3D(0,B,domain)
T_rhs = ball.TensorField_3D(0,B,domain)

# initial condition
T['g'] = 0.5*(1-r**2) + 0.1/8.*np.sqrt(35/np.pi)*r**3*(1-r**2)*(np.cos(3*phi)+np.sin(3*phi))*np.sin(theta)**3

# build state vector
state_vector = StateVector(u,p,T)
NL = StateVector(u,p,T)
#timestepper = timesteppers.SBDF4(StateVector, u,p,T)
timestepper = timesteppers.SBDF2(StateVector, u,p,T)

# build matrices
M,L,P,LU = [],[],[],[]
for ell in range(ell_start,ell_end+1):
    N = B.N_max - B.N_min(ell-B.R_max)
    M_ell,L_ell = matrices(N,ell,Ekman,Prandtl,Rayleigh)
    M.append(M_ell.astype(np.complex128))
    L.append(L_ell.astype(np.complex128))
    P.append(M_ell.astype(np.complex128))
    LU.append([None])

# calculate RHS terms from state vector
def nonlinear(state_vector, RHS, t):

    # get U in coefficient space
    state_vector.unpack(u,p,T)

    if accelerate_layout:
        om.layout = 'c'
        DT.layout = 'c'

    # take derivatives
    for ell in range(ell_start,ell_end+1):
        ell_local = ell - ell_start
        B.curl(ell,1,u['c'][ell_local],om['c'][ell_local])
        DT['c'][ell_local] = B.grad(ell,0,T['c'][ell_local])

    if accelerate_layout:
        u_rhs.layout = 'g'
        T_rhs.layout = 'g'

    # R = ez cross u
    ez = np.array([np.cos(theta),-np.sin(theta),0*np.cos(theta)])
    u_rhs['g'] = -B.cross_grid(ez,u['g'])
    u_rhs['g'] += Ekman*B.cross_grid(u['g'],om['g'])
    u_rhs['g'][0] += Rayleigh*r*T['g'][0]
    T_rhs['g'] = S - Prandtl*(u['g'][0]*DT['g'][0] + u['g'][1]*DT['g'][1] + u['g'][2]*DT['g'][2])

    # transform (ell, r) -> (ell, N)
    for ell in range(ell_start, ell_end+1):
        ell_local = ell - ell_start

        N = N_max - B.N_min(ell-R_max)

        # multiply by conversion matrices (may be very important)
        # note that M matrices are no longer conversion matrices -- need to divide by Ekman or Prandtl
        u_len = u_rhs['c'][ell_local].shape[0]
        u_rhs['c'][ell_local] = M[ell_local][:u_len,:u_len].dot(u_rhs['c'][ell_local])/Ekman
        p_len = p_rhs['c'][ell_local].shape[0]
        T_rhs['c'][ell_local] = M[ell_local][u_len+p_len:u_len+2*p_len,u_len+p_len:u_len+2*p_len].dot(T_rhs['c'][ell_local])/Prandtl

    NL.pack(u_rhs,p_rhs,T_rhs)

reducer = GlobalArrayReducer(domain.dist.comm_cart)

timing_iter = 10
if args['--run_time_iter'] is None:
    iter_end = np.inf
else:
    iter_end = int(args['--run_time_iter'])+timing_iter

out_cadence = 1000
report_cadence = 1

def initial_iterations(timing_iter):
    t = 0.
    iter = 0

    while iter <= timing_iter:
            nonlinear(state_vector,NL,t)
            if iter % report_cadence == 0:
                E0 = np.sum(weight_r*weight_theta*0.5*u['g']**2)*(np.pi)/((L_max+1)*L_dealias)
                E0 = reducer.reduce_scalar(E0, MPI.SUM)
                logger.info("iter: {:d}, dt={:e}, t/t_e={:e}, E0={:}".format(iter, dt, t/t_end, E0))

            timestepper.step(dt, state_vector, B, L, M, P, NL, LU)
            t += dt
            iter += 1
    return iter, t

logger.info("initializing")
iter, t  = initial_iterations(timing_iter)

logger.info("starting main loop")
start_time = time.time()

def main_loop(iter, t):
    try:
        # timestepping loop
        while iter <= iter_end and t <= t_end:

            if iter == timing_iter:
                start_time = time.time()

            nonlinear(state_vector,NL,t)
            if iter % report_cadence == 0:
                E0 = np.sum(weight_r*weight_theta*0.5*u['g']**2)*(np.pi)/((L_max+1)*L_dealias)
                E0 = reducer.reduce_scalar(E0, MPI.SUM)
                logger.info("iter: {:d}, dt={:e}, t/t_e={:e}, E0={:}".format(iter, dt, t/t_end, E0))

            timestepper.step(dt, state_vector, B, L, M, P, NL, LU)
            t += dt
            iter += 1
    except:
        logger.info("terminated with error")
        raise
    finally:
        logger.info("terminated normally")
    return iter, t

iter, t = main_loop(iter, t)

end_time = time.time()
if rank==0:
    n_iter_timing = iter - timing_iter
    startup_time = start_time - first_time
    main_loop_time = end_time - start_time
    logger.info('simulation took: {:.2g} sec'.format(end_time-first_time))
    logger.info('        startup: {:.2g} sec'.format(startup_time))
    logger.info('      main loop: {:.2g} sec'.format(main_loop_time))
    logger.info('       at speed: {:.3g} iter/sec'.format(n_iter_timing/(end_time-start_time)))
    if n_iter_timing > 0:
        N_TOTAL_CPU = size
        print('-' * 40)
        print('    iterations:', n_iter_timing)
        print(' loop sec/iter:', main_loop_time/n_iter_timing)
        print('    average dt:', t/n_iter_timing)
        print("          N_cores, L_max, N_max, startup     main loop,   main loop/iter, main loop/iter/grid, n_cores*main loop/iter/grid")
        print('scaling:',
            ' {:4d} {:4d} {:4d}'.format(N_TOTAL_CPU,L_max,N_max),
            ' {:8.3g} {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(startup_time,
                                                            main_loop_time,
                                                            main_loop_time/n_iter_timing,
                                                            main_loop_time/n_iter_timing/((L_max+1)**2*2*(N_max+1)),
                                                            N_TOTAL_CPU*main_loop_time/n_iter_timing/((L_max+1)**2*2*(N_max+1))))
        print('-' * 40)
