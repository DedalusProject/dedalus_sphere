import ball_diffusion as bd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import publication_settings
from analytical_eigenvalues import wavenumbers,eigenvalues
import os

matplotlib.rcParams.update(publication_settings.params)

#matplotlib.rcParams['ps.fonttype'] = 42

N_max = 255

t_mar, b_mar, l_mar, r_mar = (0.07, 0.2, 0.4, 0.1)
h_plot, w_plot = (1, 1/publication_settings.golden_mean)
h_pad = 0.03
h_pad_large = 0.25
w_pad = 0.08

num = 5

h_total = t_mar + (h_plot + h_pad)*3 + h_pad_large + b_mar
w_total = l_mar + 2*w_plot+ w_pad + r_mar

width = 8.3
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

plot_axes = []
for i in range(num):
  left = (l_mar + (w_pad + w_plot)*(i % 2)) / w_total
  bottom = 1 - (t_mar + (h_plot+h_pad)*(i//2+1) ) / h_total
  if i == 4:
    left += (w_plot + w_pad)/2/w_total
    bottom -= h_pad_large/h_total
  width = w_plot / w_total
  height = h_plot / h_total
  plot_axes.append(fig.add_axes([left, bottom, width, height]))

for i,plot_axis in enumerate(plot_axes):
  plot_axis.set_ylim([1e-16,1e0])
  plot_axis.set_xlim([0,(N_max+1)*2])
  if i < num-3:
    plt.setp(plot_axis.get_xticklabels(), visible=False)
  else:
    plot_axis.set_xlabel(r'${\rm eigenvalue} \ {\rm number}$')
  if i % 2 == 1:
    plt.setp(plot_axis.get_yticklabels(), visible=False)

eigs = []

text_label = [ r'${\rm no-slip}$',
               r'${\rm stress-free}$',
               r'${\rm potential}$',
               r'${\rm perfectly-conducting}$',
               r'${\rm pseudo-vacuum}$']

text_x = 0.025
text_y = 0.875

ell = 50

calculate = True

if not os.path.isdir('data'):
  os.mkdir('data')
if not os.path.isdir('data/eigenvalues_256'):
  os.mkdir('data/eigenvalues_256')

boundary_conditions = 'no-slip'
if calculate:
  vals, vecs = bd.eigensystem(N_max,ell,alpha_BC=2,boundary_conditions=boundary_conditions)
  np.save('data/eigenvalues_256/no_slip_eigs_a2.npy',vals)
  vals_analytic = wavenumbers(ell,N_max+1,boundary_conditions)
  vals_analytic = eigenvalues(vals_analytic,len(vals))
  np.save('data/eigenvalues_256/no_slip_eigs_analytic_a2.npy',vals_analytic)
  vals, vecs =     bd.eigensystem(N_max,ell,alpha_BC=0,boundary_conditions=boundary_conditions)
  np.save('data/eigenvalues_256/no_slip_eigs_a0.npy',vals)
  vals_analytic = wavenumbers(ell,N_max+1,boundary_conditions)
  vals_analytic = eigenvalues(vals_analytic,len(vals))
  np.save('data/eigenvalues_256/no_slip_eigs_analytic_a0.npy',vals_analytic)
  print('done with no-slip')

vals = np.load('data/eigenvalues_256/no_slip_eigs_a0.npy')
vals_analytic = np.load('data/eigenvalues_256/no_slip_eigs_analytic_a0.npy')
eigs.append(np.abs(vals**2-vals_analytic**2)/np.abs(vals_analytic**2))
vals = np.load('data/eigenvalues_256/no_slip_eigs_a2.npy')
vals_analytic = np.load('data/eigenvalues_256/no_slip_eigs_analytic_a2.npy')
eigs.append(np.abs(vals**2-vals_analytic**2)/np.abs(vals_analytic**2))

boundary_conditions = 'stress-free'
if calculate:
  vals, vecs = bd.eigensystem(N_max,ell,alpha_BC=2,boundary_conditions=boundary_conditions)
  np.save('data/eigenvalues_256/stress_free_eigs_a2.npy',vals)
  vals_analytic = wavenumbers(ell,N_max+1,boundary_conditions)
  vals_analytic = eigenvalues(vals_analytic,len(vals))
  np.save('data/eigenvalues_256/stress_free_eigs_analytic_a2.npy',vals_analytic)
  vals, vecs =     bd.eigensystem(N_max,ell,alpha_BC=0,boundary_conditions=boundary_conditions)
  np.save('data/eigenvalues_256/stress_free_eigs_a0.npy',vals)
  vals_analytic = wavenumbers(ell,N_max+1,boundary_conditions)
  vals_analytic = eigenvalues(vals_analytic,len(vals))
  np.save('data/eigenvalues_256/stress_free_eigs_analytic_a0.npy',vals_analytic)
  print('done with stress-free')

vals = np.load('data/eigenvalues_256/stress_free_eigs_a0.npy')
vals_analytic = np.load('data/eigenvalues_256/stress_free_eigs_analytic_a0.npy')
eigs.append(np.abs(vals**2-vals_analytic**2)/np.abs(vals_analytic**2))
vals = np.load('data/eigenvalues_256/stress_free_eigs_a2.npy')
vals_analytic = np.load('data/eigenvalues_256/stress_free_eigs_analytic_a2.npy')
eigs.append(np.abs(vals**2-vals_analytic**2)/np.abs(vals_analytic**2))

boundary_conditions = 'potential-field'
if calculate:
  vals, vecs = bd.eigensystem(N_max,ell,alpha_BC=2,boundary_conditions=boundary_conditions)
  np.save('data/eigenvalues_256/potential_eigs_a2.npy',vals)
  vals_analytic = wavenumbers(ell,N_max+1,'potential')
  vals_analytic = eigenvalues(vals_analytic,len(vals))
  np.save('data/eigenvalues_256/potential_eigs_analytic_a2.npy',vals_analytic)
  vals, vecs =     bd.eigensystem(N_max,ell,alpha_BC=0,boundary_conditions=boundary_conditions)
  np.save('data/eigenvalues_256/potential_eigs_a0.npy',vals)
  vals_analytic = wavenumbers(ell,N_max+1,'potential')
  vals_analytic = eigenvalues(vals_analytic,len(vals))
  np.save('data/eigenvalues_256/potential_eigs_analytic_a0.npy',vals_analytic)
  print('done with potential')

vals = np.load('data/eigenvalues_256/potential_eigs_a0.npy')
vals_analytic = np.load('data/eigenvalues_256/potential_eigs_analytic_a0.npy')
eigs.append(np.abs(vals**2-vals_analytic**2)/np.abs(vals_analytic**2))
vals = np.load('data/eigenvalues_256/potential_eigs_a2.npy')
vals_analytic = np.load('data/eigenvalues_256/potential_eigs_analytic_a2.npy')
eigs.append(np.abs(vals**2-vals_analytic**2)/np.abs(vals_analytic**2))

boundary_conditions = 'perfectly-conducting'
if calculate:
  vals, vecs = bd.eigensystem(N_max,ell,alpha_BC=2,boundary_conditions=boundary_conditions)
  np.save('data/eigenvalues_256/conducting_eigs_a2.npy',vals)
  vals_analytic = wavenumbers(ell,N_max+1,'conducting')
  vals_analytic = eigenvalues(vals_analytic,len(vals))
  np.save('data/eigenvalues_256/conducting_eigs_analytic_a2.npy',vals_analytic)
  vals, vecs =     bd.eigensystem(N_max,ell,alpha_BC=0,boundary_conditions=boundary_conditions)
  np.save('data/eigenvalues_256/conducting_eigs_a0.npy',vals)
  vals_analytic = wavenumbers(ell,N_max+1,'conducting')
  vals_analytic = eigenvalues(vals_analytic,len(vals))
  np.save('data/eigenvalues_256/conducting_eigs_analytic_a0.npy',vals_analytic)
  print('done with conducting')

vals = np.load('data/eigenvalues_256/conducting_eigs_a0.npy')
vals_analytic = np.load('data/eigenvalues_256/conducting_eigs_analytic_a0.npy')
eigs.append(np.abs(vals**2-vals_analytic**2)/np.abs(vals_analytic**2))
vals = np.load('data/eigenvalues_256/conducting_eigs_a2.npy')
vals_analytic = np.load('data/eigenvalues_256/conducting_eigs_analytic_a2.npy')
eigs.append(np.abs(vals**2-vals_analytic**2)/np.abs(vals_analytic**2))

boundary_conditions = 'pseudo-vacuum'
if calculate:
  vals, vecs = bd.eigensystem(N_max,ell,alpha_BC=2,boundary_conditions=boundary_conditions)
  np.save('data/eigenvalues_256/pseudo_vacuum_eigs_a2.npy',vals)
  vals_analytic = wavenumbers(ell,N_max+1,'pseudo')
  vals_analytic = eigenvalues(vals_analytic,len(vals))
  np.save('data/eigenvalues_256/pseudo_vacuum_eigs_analytic_a2.npy',vals_analytic)
  vals, vecs =     bd.eigensystem(N_max,ell,alpha_BC=0,boundary_conditions=boundary_conditions)
  np.save('data/eigenvalues_256/pseudo_vacuum_eigs_a0.npy',vals)
  vals_analytic = wavenumbers(ell,N_max+1,'pseudo')
  vals_analytic = eigenvalues(vals_analytic,len(vals))
  np.save('data/eigenvalues_256/pseudo_vacuum_eigs_analytic_a0.npy',vals_analytic)
  print('done with pseudo-vacuum')

vals = np.load('data/eigenvalues_256/pseudo_vacuum_eigs_a0.npy')
vals_analytic = np.load('data/eigenvalues_256/pseudo_vacuum_eigs_analytic_a0.npy')
eigs.append(np.abs(vals**2-vals_analytic**2)/np.abs(vals_analytic**2))
vals = np.load('data/eigenvalues_256/pseudo_vacuum_eigs_a2.npy')
vals_analytic = np.load('data/eigenvalues_256/pseudo_vacuum_eigs_analytic_a2.npy')
eigs.append(np.abs(vals**2-vals_analytic**2)/np.abs(vals_analytic**2))

for i in range(5):
  plot_axes[i].semilogy(eigs[2*i+1],linewidth=2,color='MidnightBlue',label=r'$\alpha_{BC}=2$')
  plot_axes[i].semilogy(eigs[2*i],linewidth=2,color='FireBrick',label=r'$\alpha_{BC}=0$')
  lg = plot_axes[i].legend(loc='lower right',fontsize=12)
  lg.draw_frame(False)
  if i % 2 == 0:
    plot_axes[i].set_ylabel(r'$\frac{\left|\lambda-\lambda_a\right|}{\left|\lambda_a\right|}$',fontsize=14)
  plot_axes[i].text(text_x, text_y, text_label[i],
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=14,
                    transform = plot_axes[i].transAxes)
  if i>0 and i < 4:
    plot_axes[i].yaxis.set_ticks([1e-16,1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2])

plt.savefig('figures/eigenvalues_BC.png')

