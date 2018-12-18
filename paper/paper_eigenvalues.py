import ball_bessel       as bessel
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import publication_settings
from analytical_eigenvalues import wavenumbers
import sphere_rossby as rossby
import time

matplotlib.rcParams.update(publication_settings.params)

#matplotlib.rcParams['ps.fonttype'] = 42

N_max = 511

t_mar, b_mar, l_mar, r_mar = (0.1, 0.25, 0.5, 0.1)
h_plot, w_plot = (1, 1/publication_settings.golden_mean)
w_pad = 0.55

num = 2
h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot*num + w_pad*(num-1) + r_mar

width = 7.1
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

plot_axes = []
for i in range(num):
  left = (l_mar + i*(w_plot+w_pad) ) / w_total
  bottom = 1 - (t_mar + h_plot ) / h_total
  width = w_plot / w_total
  height = h_plot / h_total
  plot_axes.append(fig.add_axes([left, bottom, width, height]))

for i,plot_axis in enumerate(plot_axes):
  plot_axis.set_ylim([1e-16,1])
  plot_axis.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10,numticks=8))
  plot_axis.set_xlim([0,N_max+1])
  plot_axis.set_xlabel(r'${\rm eigenvalue} \ {\rm number}$')


eigs = []
labels = [r'$\frac{|\omega-\omega_a|}{|\omega_a|}$',
          r'$\frac{|\kappa-\kappa_a|}{|\kappa_a|}$']

text_label = [r'${\rm Rossby}$',
               r'${\rm Bessel}$']

text_x = 0.2
text_y = 0.875

calculate = True

if calculate:
  rossby_start = time.time()
  m = 50
  vals, th, vth , vph, p = rossby.eigensystem(N_max+m,m)
  np.save('data/rossby.npy',vals)
  ell = np.arange(m,len(vals)+m)
  vals_analytic = -m/(ell*(ell+1))
  np.save('data/rossby_analytic.npy',vals_analytic)
  rossby_end = time.time()
  print(vals)
  print(vals_analytic)
  print("Rossby eigenvalues took {:g} sec".format(rossby_end-rossby_start))

vals = np.load('data/rossby.npy')
vals_analytic = np.load('data/rossby_analytic.npy')

eigs.append(np.abs(vals - vals_analytic)/np.abs(vals_analytic))
N_bessel_set = [N_max, int(3/2*N_max)]

for N_bessel in N_bessel_set:
  if calculate:
    bessel_start = time.time()
    ell = 50
    vals, r, vec = bessel.eigensystem(N_bessel,ell,cutoff=np.inf)
    np.save('data/bessel_{}.npy'.format(N_bessel),np.sqrt(vals))
    vals_analytic = wavenumbers(ell,N_bessel,"Bessel")
    np.save('data/bessel_analytic_{}.npy'.format(N_bessel),vals_analytic)
    bessel_end = time.time()
    print("Bessel eigenvalues took {:g} sec".format(bessel_end-bessel_start))
    print("eigenvalues found: analytic {} vs numeric {}".format(vals_analytic.shape, vals.shape))

  vals = np.load('data/bessel_{}.npy'.format(N_bessel))
  vals_analytic = np.load('data/bessel_analytic_{}.npy'.format(N_bessel))

  eigs.append(np.abs(vals - vals_analytic)/np.abs(vals_analytic))

color=['MidnightBlue', 'MidnightBlue', 'FireBrick']
for i in range(2):
  if i == 1:
    plot_axes[i].semilogy(eigs[i+1],linewidth=2,color=color[i+1])
  plot_axes[i].semilogy(eigs[i],linewidth=2,color=color[i])
  plot_axes[i].set_ylabel(labels[i])
  plot_axes[i].text(text_x, text_y, text_label[i],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    transform = plot_axes[i].transAxes)

plt.savefig('figures/eigenvalues.eps')

