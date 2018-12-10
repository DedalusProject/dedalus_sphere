import matplotlib
import matplotlib.pyplot as plt
import numpy             as np
import publication_settings

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.05, 0.18, 0.4, 0.07)
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

data = np.loadtxt('data/marti_hydro_E_32_32_a2_CNAB2_0p01.dat')

t = data[0,:]
E = data[1,:]

plot_axes.plot(t,E,linewidth=2,color='MidnightBlue')

plot_axes.set_xlabel(r'$t$')
plot_axes.set_ylabel(r'$\frac{1}{2}\int \ |\mathbf{u}|^2 \ dV$')

plt.savefig('figures/hydro_E.eps')

