import shell_bessel       as bessel
from dedalus_sphere import annulus as shell
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker   import LogLocator
import numpy             as np
import scipy.special     as spec
import publication_settings

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.05, 0.2, 0.27, 0.07)
h_eig, w_eig = (1,3)
h_pad = 0.075
h_error, w_error = (0.25,3)

h_total = t_mar + h_eig + h_pad + h_error + b_mar
w_total = l_mar + w_eig + r_mar

width = 7.1
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

left = l_mar / w_total
bottom = 1 - (t_mar + h_eig) / h_total
width = w_eig / w_total
height = h_eig / h_total
eig_axes = fig.add_axes([left,bottom,width,height])

#inset_width = 0.2
#inset_height = 0.25
#inset_left = 0.77
#inset_bottom = 0.7
#
#left = left + width*inset_left
#bottom = bottom + height*inset_bottom
#width = width*inset_width
#height = height*inset_height
#
#inset_axes = fig.add_axes([left,bottom,width,height])

left = l_mar / w_total
bottom = 1 - (t_mar + h_eig + h_pad + h_error) / h_total
width = w_error / w_total
height = h_error / h_total
error_axes = fig.add_axes([left,bottom,width,height])

N_max = 511
ell = 50
eig_num = 100
radii = Ri, Ro = (1, 2)

vals, r, vec = bessel.eigensystem(N_max,ell,radii,cutoff=np.inf)

z, w = shell.quadrature(2047, niter=3)
r = ( (Ro - Ri) * z + (Ro + Ri) )/2
r = r.astype(np.float64)

Q = shell.trial_functions(N_max, z)
f = (Q.T) @ vec[eig_num]
f = f.real
f /= np.max(np.abs(f))
if np.max(-f) > np.max(f): f *= -1

k = np.sqrt(vals[eig_num])
sol = spec.jv(ell+1/2,k*r)/np.sqrt(k*r)*spec.yv(ell+1/2,k) - spec.yv(ell+1/2,k*r)/np.sqrt(k*r)*spec.jv(ell+1/2,k)
sol /= np.max(np.abs(sol))
if np.max(-sol) > np.max(sol): sol *= -1

eig_axes.plot(r,f,color='firebrick',linewidth=2,label=r'${\rm numeric}$')
eig_axes.plot(r,sol,color='midnightblue',linewidth=1,label=r'${\rm numeric}$')
eig_axes.set_ylabel(r'$f$')
plt.setp(eig_axes.get_xticklabels(), visible=False)
eig_axes.set_ylim([-1.1,1.4])
eig_axes.set_xlim([1,2])
eig_axes.set_yticks([-1.,0,1.])

#inset_axes.loglog(r,sol,color='firebrick')
#inset_axes.loglog(r,1e10*sol[0]*(r/r[0])**(ell),color='k',linestyle='--')
#inset_axes.set_xlim([0,0.1])
#inset_axes.set_ylim([np.min(sol),1.])
#inset_axes.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
#inset_axes.set_xticks([0.01,0.1])
#inset_axes.set_ylabel(r'$f$')
#inset_axes.set_xlabel(r'$r$',labelpad = -1)
#
#inset_axes.annotate(
#              r'$r^{50}$',
#              xy=(3e-3, 1e-50), xytext=(3e-3, 1e-50),
#              fontsize=12,
#              textcoords='offset points')

error_axes.plot(r,(sol-f),color='k',linewidth=2)
#error_axes.set_ylabel(r'${\rm error}\times 10^{13}$')
error_axes.set_xlabel(r'$r$')
#error_axes.set_yticks([-3.,0,3])
error_axes.set_xlim([1,2])

plt.savefig('figures/bessel_eigenfunction.png')


