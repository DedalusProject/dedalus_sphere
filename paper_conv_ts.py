import matplotlib
import matplotlib.pyplot as plt
import numpy             as np
import publication_settings

from load_data_conv import *

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.07, 0.23, 0.3, 0.07)
h_plot, w_plot = (1,1/publication_settings.golden_mean)

h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + r_mar

width = 4.3
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

left = l_mar / w_total
bottom = 1 - (t_mar + h_plot) / h_total
width = w_plot / w_total
height = h_plot / h_total
plot_axes = fig.add_axes([left,bottom,width,height])

dt = np.array([8e-5,4e-5,2e-5,1e-5])
dE = np.array([E_32_a2_C_8en5,E_32_a2_C_4en5,E_32_a2_C_2en5,E_32_a2_C_1en5]) - E_32_a2_S_1en5

plot_axes.loglog(dt,dE,linewidth=2,color='MidnightBlue',marker='x',markersize=10,markeredgewidth=2)
plot_axes.loglog(dt,(dt/dt[0])**(2)*1e-2,color='Firebrick',linestyle='--')

plot_axes.set_xlabel(r'${\rm timestep \ size}$')
plot_axes.set_ylabel(r'$KE-KE_c$')
plot_axes.set_xlim([0.8e-5,1e-4])
plot_axes.set_ylim([5e-5,1.5e-2])

plot_axes.annotate(
             r'$\Delta t^2$',
             xy=(2e-5, 1.2e-3), xytext=(2e-5, 1.2e-3),
             fontsize=12,
             textcoords='offset points')

plt.savefig('figures/conv_ts.eps')

