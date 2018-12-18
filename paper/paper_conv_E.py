import matplotlib
import matplotlib.pyplot as plt
import numpy             as np
import publication_settings

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.05, 0.18, 0.2, 0.07)
h_plot, w_plot = (1,1/publication_settings.golden_mean)
w_pad = 0.3

h_total = t_mar + h_plot + b_mar
w_total = l_mar + 2*w_plot + w_pad + r_mar

width = 8.3
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

plot_axes = []

for i in range(2):
  left = (l_mar + (w_pad + w_plot)*i ) / w_total
  bottom = 1 - (t_mar + h_plot) / h_total
  width = w_plot / w_total
  height = h_plot / h_total
  plot_axes.append(fig.add_axes([left,bottom,width,height]))

data = np.loadtxt('data/marti_conv_E_32_32_a2_SBDF4_2en5.dat')
t_a2_2en5 = data[0,:]
E_a2_2en5 = data[1,:]

data = np.loadtxt('data/marti_conv_E_32_32_a2_SBDF4_1en5.dat')
t_a2_1en5 = data[0,:]
E_a2_1en5 = data[1,:]

data = np.loadtxt('data/marti_conv_E_32_32_a2_SBDF4_2en5.dat')
t_a2_2en5 = data[0,:]
E_a2_2en5 = data[1,:]

data = np.loadtxt('data/marti_conv_E_32_32_a0_SBDF4_2en5.dat')
t_a0_2en5 = data[0,:]
E_a0_2en5 = data[1,:]

data = np.loadtxt('data/marti_conv_E_32_32_a0_SBDF4_1en5.dat')
t_a0_1en5 = data[0,:]
E_a0_1en5 = data[1,:]

plot_axes[0].plot(t_a2_1en5,E_a2_1en5,linewidth=2,color='MidnightBlue',label=r'$\alpha_{BC}=2,\,\Delta t=10^{-5}$')

plot_axes[0].set_xlabel(r'$t$')
plot_axes[0].set_ylabel(r'$KE$')
tick_locs = [0,5,10,15,20]
plot_axes[0].set_xticklabels([r"$%s$" % x for x in tick_locs])
ytick_locs = [0,5,10,15,20,25,30,35]
plot_axes[0].set_yticklabels([r"$%s$" % x for x in ytick_locs])
lg = plot_axes[0].legend(loc='upper right',fontsize=10)
lg.draw_frame(False)

E_c = 29.12045489

plot_axes[1].plot(t_a0_1en5,E_a0_1en5-E_c,linewidth=2,color='MidnightBlue',label=r'$\alpha_{BC}=0,\,\Delta t=10^{-5}$')
plot_axes[1].plot(t_a0_2en5,E_a0_2en5-E_c,linewidth=2,color='ForestGreen',label=r'$\alpha_{BC}=0,\,\Delta t=2\times 10^{-5}$')
plot_axes[1].plot(t_a2_1en5,E_a2_1en5-E_c,linewidth=2,color='DarkGoldenrod',label=r'$\alpha_{BC}=2,\,\Delta t=10^{-5}$')
plot_axes[1].plot(t_a2_2en5,E_a2_2en5-E_c,linewidth=2,color='FireBrick',label=r'$\alpha_{BC}=2,\,\Delta t=2\times 10^{-5}$')

plot_axes[1].set_ylim([-1e-8,1e-8])
plot_axes[1].set_ylabel(r'$KE-KE_c$',labelpad=-10)
plot_axes[1].set_xlabel(r'$t$')
plot_axes[1].set_xticklabels([r"$%s$" % x for x in tick_locs])
plot_axes[1].yaxis.set_ticks([-1e-8,0,1e-8])
plot_axes[1].set_yticklabels([r'$-10^{-8}$',r'$0$',r'$10^{-8}$'])
lg = plot_axes[1].legend(loc='upper right',fontsize=10)
lg.draw_frame(False)

plt.savefig('figures/conv_E.eps')

