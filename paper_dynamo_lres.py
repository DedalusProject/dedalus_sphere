import matplotlib
import matplotlib.pyplot as plt
import numpy             as np
import publication_settings
from matplotlib.ticker import FormatStrFormatter,FuncFormatter
from matplotlib.legend import Legend

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.07, 0.22, 0.35, 0.07)
h_plot, w_plot = (1,1/publication_settings.golden_mean)

h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + r_mar

width = 4.3
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

plot_axes = []

left = (l_mar) / w_total
bottom = 1 - (t_mar + h_plot) / h_total
width = w_plot / w_total
height = h_plot / h_total
plot_axes.append(fig.add_axes([left,bottom,width,height]))

data_a2   = np.loadtxt('data/marti_dynamo_E_24_24_a2_CNAB2_5en6.dat')
data_a1p5 = np.loadtxt('data/marti_dynamo_E_24_24_a1p5_CNAB2_5en6.dat')
data_a0   = np.loadtxt('data/marti_dynamo_E_24_24_a0_CNAB2_5en6.dat')

t_a2 = data_a2[:,0]
ME_a2 = data_a2[:,3]/1.5

t_a1p5 = data_a1p5[:,0]
ME_a1p5 = data_a1p5[:,3]

t_a0 = data_a0[:,0]
ME_a0 = data_a0[:,3]

def format_fn(tick_val, tick_pos):
    if int(tick_val)//10000 == 0: return r"$0$"
    if int(tick_val)//10000 == 1: return r"$10^4$"
    return r"$%i\times 10^4$" %(int(tick_val)//10000)

ME_correct=905.566

labels = [r'$\alpha_{BC}=2$',r'$\alpha_{BC}=3/2$',r'$\alpha_{BC}=0$',r'$\overline{ME}$']

l0, = plot_axes[0].plot(t_a2,ME_a2,linewidth=2,color='MidnightBlue')
l1, = plot_axes[0].plot(t_a1p5,ME_a1p5,linewidth=2,color='DarkGoldenrod')
l2, = plot_axes[0].plot(t_a0,ME_a0,linewidth=2,color='FireBrick')
l3, = plot_axes[0].plot([t_a0[0],t_a0[-1]],[ME_correct,ME_correct],color='k')
lg = plot_axes[0].legend([l0,l1,l2],labels[:3],loc='upper left',fontsize=10)
lg.draw_frame(False)
lg2 = Legend(plot_axes[0], [l3], [labels[-1]],loc='upper right',fontsize=10, frameon=False)
plot_axes[0].add_artist(lg2)
plot_axes[0].set_xlabel(r'$t$')
plot_axes[0].set_ylabel(r'$ME$')
plot_axes[0].set_ylim([0,1800])
plot_axes[0].set_xlim([0,10])
#tick_locs = [0,1,2,3,4,5,6]
#plot_axes[0].set_xticklabels([r"$%s$" % x for x in tick_locs])

plt.savefig('figures/dynamo_lres.eps')

