import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import publication_settings
import pickle
from dedalus.extras import plot_tools
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams.update(publication_settings.params)

#matplotlib.rcParams['ps.fonttype'] = 42

t_mar, b_mar, l_mar, r_mar = (0.1, 0.05, 0.05, 0.05)
h_plot, w_plot = (1, 1)
w_pad = 0.08
h_pad = 0.1
h_cbar = 0.03

num = 4

h_total = t_mar + h_plot + h_pad + h_cbar + b_mar
w_total = l_mar + 4*w_plot+ 3*w_pad + r_mar

width = 8.3
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

plot_axes = []
for i in range(num):
  left = (l_mar + (w_pad + w_plot)*(i)) / w_total
  bottom = 1 - (t_mar + h_plot + h_pad + h_cbar ) / h_total
  width = w_plot / w_total
  height = h_plot / h_total
  plot_axes.append(fig.add_axes([left, bottom, width, height]))

cbar_axes = []
for i in range(num):
  left = (l_mar + (w_pad + w_plot)*(i)) / w_total
  bottom = 1 - (t_mar + h_cbar ) / h_total
  width = w_plot / w_total
  height = h_cbar / h_total
  cbar_axes.append(fig.add_axes([left, bottom, width, height]))

data = pickle.load(open('data/marti_hydro_mid.pkl','rb'))

r = data['r']
phi = data['phi']
ur = data['ur']
uphi = data['uph']

phi_shape = phi.shape
r_shape = r.shape
phi = phi.reshape((phi_shape[0],1))
r = r.reshape((1,r_shape[-1]))
phi_1D = np.concatenate((phi[:,0],[0]))

plot_data = []
plot_data.append(ur) # ur
plot_data.append(uphi) # uphi
plot_data.append(ur*np.cos(phi) - uphi*np.sin(phi)) # ux
plot_data.append(ur*np.sin(phi) + uphi*np.cos(phi)) # uy

labels = [r'$u_r$',r'$u_\phi$',r'$u_x$',r'$u_y$']

r, phi = plot_tools.quad_mesh(r.ravel(), phi.ravel())
r[:, 0] = 0

x = r*np.cos(phi)
y = r*np.sin(phi)

c_im = []
cbars = []

eps = 0.02

for i in range(num):
#  c_im.append(plot_axes[i].pcolormesh(r,phi,plot_data[i]))
  c_im.append(plot_axes[i].pcolormesh(x,y,plot_data[i],cmap='RdBu'))
  plot_axes[i].plot((1+eps/2)*np.sin(phi),(1+eps/2)*np.cos(phi),color='k',linewidth=2)
  plot_axes[i].set_axis_off()
  plot_axes[i].axis([-1-2*eps,1+2*eps,-1-2*eps,1+2*eps])
  nbins = 4
  if i==2: nbins = 5
  cbars.append(fig.colorbar(c_im[i], cax = cbar_axes[i], orientation='horizontal', ticks=MaxNLocator(nbins=nbins)))
  cbars[i].ax.tick_params(labelsize=8)
  cbar_axes[i].text(0.5,3,labels[i],va='center',ha='center',fontsize=12,transform=cbar_axes[i].transAxes)


plt.savefig('figures/equatorial.png', dpi=600)


