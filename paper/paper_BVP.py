
from dedalus_sphere import ball_wrapper as ball
import numpy as np
from scipy.sparse import linalg as spla
import scipy.sparse      as sparse
import dedalus.public as de
from dedalus.core.distributor import Distributor

def err(a,b):
    print(np.max(np.abs(a-b))/np.max(np.abs(b)))

L_max, N_max, R_max = 31, 31, 2
B_3D = ball.Ball(N_max,L_max,R_max=R_max)

L_dealias = 1
N_dealias = 1
N_r = N_max

comm = None
mesh = None
phi_basis_3D = de.Fourier('phi',2*(L_max+1), interval=(0,2*np.pi),dealias=L_dealias)
theta_basis_3D = de.Fourier('theta', L_max+1, interval=(0,np.pi),dealias=L_dealias)
r_basis_3D = de.Fourier('r', N_max+1, interval=(0,1),dealias=N_dealias)
domain_3D = de.Domain([phi_basis_3D,theta_basis_3D,r_basis_3D],
                      grid_dtype=np.float64,comm=comm,mesh=mesh)

domain_3D.global_coeff_shape = np.array([L_max+1,L_max+1,N_max+1])
domain_3D.distributor = Distributor(domain_3D,comm,mesh)

phi_layout   = domain_3D.distributor.layouts[3]
grid_slices = phi_layout.slices(domain_3D.dealias)
phi = domain_3D.grid(0,scales=domain_3D.dealias)
theta = B_3D.grid(1,dimensions=3)[:,grid_slices[1],:] # local
r = B_3D.grid(2,dimensions=3)[:,:,grid_slices[2]] # local

H = ball.TensorField_3D(1,B_3D,domain_3D)
H2= ball.TensorField_3D(1,B_3D,domain_3D)
A = ball.TensorField_3D(1,B_3D,domain_3D)

H['g'][1] = -3./2.*r*(-1+4*r**2-6*r**4+3*r**6)*(np.cos(phi)+np.sin(phi))
H['g'][2] = -3./4.*r*(-1+r**2)*np.cos(theta)* \
                 ( 3*r*(2-5*r**2+4*r**4)*np.sin(theta)
                  +2*(1-3*r**2+3*r**4)*(np.cos(phi)-np.sin(phi)))

for ell in range(1,L_max+1):
    N  = N_max - B_3D.N_min(ell-R_max)

    Z = B_3D.op('0',N,0,ell)

    xim = B_3D.xi(-1,ell)
    xip = B_3D.xi(+1,ell)

    L00 = Z
    L02 = Z

    Dm = B_3D.op('D-',N,0,ell).astype(np.complex128)
    L01 = -1j*xip*Dm

    Dm_p = B_3D.op('D-',N,0,ell+1).astype(np.complex128)
    Dp_m = B_3D.op('D+',N,0,ell-1).astype(np.complex128)

    L10 = -1j*xip*Dp_m
    L11 = Z
    L12 = 1j*xim*Dm_p

    L20 = xim*Dp_m
    L21 = Z
    L22 = xip*Dm_p

    L=sparse.bmat([[L00, L01, L02],
                   [L10, L11, L12],
                   [L20, L21, L22]])

    L = L.tocsr()

    L[N]=np.concatenate((B_3D.op('r=1',N,0,ell-1),np.zeros(2*(N+1))))
    L[2*N+1]=np.concatenate((np.zeros(N+1),B_3D.op('r=1',N,1,ell-1)*Dm,np.zeros(N+1)))
    L[3*N+2]=np.concatenate((np.zeros(2*(N+1)),B_3D.op('r=1',N,1,ell  )*Dm_p))

    RHS = np.copy(H['c'][ell])
    Em = B_3D.op('E',N,0,ell-1).astype(np.complex128)
    E0 = B_3D.op('E',N,0,ell).astype(np.complex128)
    RHS[:(N+1)] = Em.dot(RHS[:(N+1)])
    RHS[(N+1):2*(N+1)] = E0.dot(RHS[(N+1):2*(N+1)])
    RHS[2*(N+1):] *= 0.

    A['c'][ell] = spla.spsolve(L,RHS)

A_analytic_0 = (3/2*r**2*(1-4*r**2+6*r**4-3*r**6)
                   *np.sin(theta)*(np.sin(phi)-np.cos(phi))
               +3/8*r**3*(2-7*r**2+9*r**4-4*r**6)
                   *(3*np.cos(theta)**2-1)
               +9/160*r**2*(-200/21*r+980/27*r**3-540/11*r**5+880/39*r**7)
                     *(3*np.cos(theta)**2-1)
               +9/80*r*(1-100/21*r**2+245/27*r**4-90/11*r**6+110/39*r**8)
                    *(3*np.cos(theta)**2-1)
               +1/8*r*(-48/5*r+288/7*r**3-64*r**5+360/11*r**7)
                   *np.sin(theta)*(np.sin(phi)-np.cos(phi))
               +1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                   *np.sin(theta)*(np.sin(phi)-np.cos(phi)))
A_analytic_1 = (-27/80*r*(1-100/21*r**2+245/27*r**4-90/11*r**6+110/39*r**8)
                        *np.cos(theta)*np.sin(theta)
                +1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                    *np.cos(theta)*(np.sin(phi)-np.cos(phi)))
A_analytic_2 = (1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                   *(np.cos(phi)+np.sin(phi)))

err(A['g'][0],A_analytic_0)
err(A['g'][1],A_analytic_1)
err(A['g'][2],A_analytic_2)

for ell in range(0,L_max+1):
    B_3D.curl(ell,1,A['c'][ell],H2['c'][ell])

err(H['g'],H2['g'])


