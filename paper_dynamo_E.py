import matplotlib
import matplotlib.pyplot as plt
import numpy             as np
import publication_settings
from matplotlib.ticker import FormatStrFormatter,FuncFormatter

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.07, 0.22, 0.35, 0.07)
w_pad = 0.35
h_plot, w_plot = (1,1/publication_settings.golden_mean)
h_inset, w_inset = (0.25*h_plot,0.5*w_plot)
l_inset = 0.47*w_plot
t_inset = 0.1*h_plot
l_inset2 = l_inset
t_inset2 = 1 - t_inset - h_inset - 0.03

h_total = t_mar + h_plot + b_mar
w_total = l_mar + 2*w_plot + w_pad + r_mar

width = 8.3
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))


plot_axes = []
inset_axes = []
inset2_axes = []

for i in range(2):
  left = (l_mar + (w_pad+w_plot)*i) / w_total
  bottom = 1 - (t_mar + h_plot) / h_total
  width = w_plot / w_total
  height = h_plot / h_total
  plot_axes.append(fig.add_axes([left,bottom,width,height]))

  left = (l_mar + l_inset + (w_pad+w_plot)*i ) / w_total
  bottom = 1 - (t_mar + h_inset + t_inset) / h_total
  width = w_inset / w_total
  height = h_inset / h_total
  inset_axes.append(fig.add_axes([left,bottom,width,height]))

  left = (l_mar + l_inset2 + (w_pad+w_plot)*i ) / w_total
  bottom = 1 - (t_mar + h_inset + t_inset2) / h_total
  width = w_inset / w_total
  height = h_inset / h_total
  inset2_axes.append(fig.add_axes([left,bottom,width,height]))

data = np.loadtxt('data/marti_dynamo_E_64_64_a2_SBDF4_2p5en6.dat')

t = data[:,0]
KE = data[:,1]
ME = data[:,2]

def format_fn(tick_val, tick_pos):
    if int(tick_val)//10000 == 0: return r"$0$"
    if int(tick_val)//10000 == 1: return r"$10^4$"
    return r"$%i\times 10^4$" %(int(tick_val)//10000)

plot_axes[0].yaxis.set_major_formatter(FuncFormatter(format_fn))
plot_axes[0].plot(t,KE,linewidth=2,color='MidnightBlue')
plot_axes[0].set_xlabel(r'$t$')
plot_axes[0].set_ylabel(r'$KE$')
#plot_axes[0].yaxis.set_major_formatter(FormatStrFormatter('%4e'))
plot_axes[0].set_ylim([0,70000])
tick_locs = [0,1,2,3,4,5,6,7,8,9]
plot_axes[0].set_xticklabels([r"$%s$" % x for x in tick_locs]) 

KE_max = 37444.32
KE_min = 33681.31

inset_times = [6,7,8,9]
inset_diffs = [-0.01,0,0.01]

inset_axes[0].plot(t,KE-KE_max,linewidth=0.5,color='MidnightBlue')
inset_axes[0].set_ylim([-1e-2,1e-2])
inset_axes[0].set_xlim([6,9])
inset_axes[0].yaxis.set_ticks([-1e-2,0,1e-2])
inset_axes[0].xaxis.set_ticks([6,7,8,9])
inset_axes[0].set_yticklabels([r"$%s$" % y for y in inset_diffs])
inset_axes[0].set_xticklabels([r"$%s$" % x for x in inset_times])
inset_axes[0].set_ylabel(r'$KE-KE_{\rm sup}$')

inset2_axes[0].plot(t,KE-KE_min,linewidth=0.5,color='MidnightBlue')
inset2_axes[0].set_ylim([-1e-2,1e-2])
inset2_axes[0].set_xlim([6,9])
inset2_axes[0].yaxis.set_ticks([-1e-2,0,1e-2])
inset2_axes[0].xaxis.set_ticks([6,7,8,9])
inset2_axes[0].set_yticklabels([r"$%s$" % y for y in inset_diffs])
inset2_axes[0].set_xticklabels([r"$%s$" % x for x in inset_times])
inset2_axes[0].set_ylabel(r'$KE-KE_{\rm inf}$')

plot_axes[1].plot(t,ME,linewidth=2,color='MidnightBlue')
plot_axes[1].set_xlabel(r'$t$')
plot_axes[1].set_ylabel(r'$ME$')
plot_axes[1].set_ylim([0,1800])
ME_locs = [0,200,400,600,800,1000,1200,1400,1600,1800]
plot_axes[1].set_yticklabels([r"$%s$" % y for y in ME_locs])
plot_axes[1].set_xticklabels([r"$%s$" % x for x in tick_locs])

ME_max = 943.4111
ME_min = 867.7413

inset_times = [6,7,8,9]
inset_diffs = [-0.0002,0,0.0002]

inset_axes[1].plot(t,ME-ME_max,linewidth=0.5,color='MidnightBlue')
inset_axes[1].set_ylim([-2e-4,2e-4])
inset_axes[1].set_xlim([6,9])
inset_axes[1].yaxis.set_ticks([-2e-4,0,2e-4])
inset_axes[1].xaxis.set_ticks([6,7,8,9])
inset_axes[1].set_yticklabels([r"$%s$" % y for y in inset_diffs])
inset_axes[1].set_xticklabels([r"$%s$" % x for x in inset_times])
inset_axes[1].set_ylabel(r'$ME-ME_{\rm sup}$')

inset2_axes[1].plot(t,ME-ME_min,linewidth=0.5,color='MidnightBlue')
inset2_axes[1].set_ylim([-2e-4,2e-4])
inset2_axes[1].set_xlim([6,9])
inset2_axes[1].yaxis.set_ticks([-2e-4,0,2e-4])
inset2_axes[1].xaxis.set_ticks([6,7,8,9])
inset2_axes[1].set_yticklabels([r"$%s$" % y for y in inset_diffs])
inset2_axes[1].set_xticklabels([r"$%s$" % x for x in inset_times])
inset2_axes[1].set_ylabel(r'$ME-ME_{\rm inf}$')

plt.savefig('figures/dynamo_E.eps')


