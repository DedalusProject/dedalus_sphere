#!/usr/bin/env python3
"""
Perform scaling runs on special scaling scripts.  This driver script
should be run serially, and it will then spawn off a series of MPI
processes to test the scaling performance of a given machine.

The target script <scaling_script> is assumed to take the following
command line inputs:

3-D scaling script (1- or 2-D processor decomposition):

   scaling_script.py --L_max=L_max -N_max=N_max --mesh=p1,p2

where n_max is the radial modal resolution, and L_max is the
spherical harmonic modal resolutions.  The mesh keyword should
accept the 2-D processor mesh, with p1 and p2 the processor mesh values.

If the 3-D scaling script is not passed the mesh keyword, it should default
to a 1-D domain decomposition.

These scaling scripts should output well formated scaling outputs,
following the example scripts.  In a future revision, that output will
be rolled into this scaling.py package.



Usage:
    scaling.py run <scaling_script> [options]
    scaling.py plot <files>... [options]

Options:
    --L_max=<L_max>        set coefficient resolution in horizontal direction [default: 255]
    --N_max=<N_max>        set coefficient resolution in radial direction; if not set, default is half of Lmax

    --label=<label>        Label for output file
    --niter=<niter>        Number of iterations to run for [default: 100]
    --verbose              Print verbose output at end of each run (stdout and stderr)

    --one-pencil           Push to one pencil per core in coeff space
    --limit-mem            Limited memory; restrict low end of core count

    --max-cores=<max-cores>      Max number of available cores
    --min-cores=<min-cores>      Min number of cores to use

    --output=<dir>         Output directory [default: ./scaling]
    --rescale=<rescale>    rescale plots to particular Z resolution comparison case
    --clean_plot           Remove run-specific labels during plotting (e.g., for proposals or papers)

    --OpenMPI              Assume we're in an OpenMPI env; default if nothing else is selected
    --MPISGI               Assume we're in a SGI-MPT env (e.g., NASA Pleiades)
    --IntelMPI             Assume we're in an IntelMPI env (e.g., PSC Bridges)
"""
import os
import numpy as np
import itertools
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
        plt.style.use('ggplot')
except:
        print("Upgrade matplotlib; for now we're falling back to old plot styles")

import time
import shelve
import pathlib

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def do_scaling_run(scaling_script, resolution, CPU_set, max_cores=None, min_cores=None,
                   niter=None,
                   test_type='exhaustive', verbose=None, label=None, dim=2,
                   OpenMPI=None, MPISGI=None, IntelMPI=None):
    if OpenMPI is None and IntelMPI is None and MPISGI is None:
        OpenMPI = True

    if dim == 3:
        import itertools
        CPU_set_1 = CPU_set
        CPU_set_2 = CPU_set

        if max_cores is not None:
            if (np.max(CPU_set_1)**2) < max_cores:
                # append new element to front of set_2
                CPU_set_2 = np.append(2*np.max(CPU_set_2), CPU_set_2)
        if min_cores is not None:
            if (np.min(CPU_set_1)*np.min(CPU_set_2)) > min_cores:
                # append new element to end of set_1
                CPU_set_1 = np.append(CPU_set_1, np.int(np.min(CPU_set_1)/2))
        print(CPU_set_1)
        print(CPU_set_2)
        print('testing from {:d} to {:d} cores'.format(np.min(CPU_set_1)*np.min(CPU_set_2),np.max(CPU_set_1)*np.max(CPU_set_2)))
        if test_type=='exhaustive':
            print('doing exhaustive scaling test')
            scaling_test_set = itertools.product(CPU_set_1, CPU_set_2)
        elif test_type=='patient':
            print('doing patient scaling test')
            scaling_test_set = itertools.combinations_with_replacement(CPU_set, 2)
        else:
            # symmetric_cobminations
            print('doing minimal scaling test')
            scaling_test_set = zip(CPU_set_1, CPU_set_2)

    else:
        print('testing {}, from {:d} to {:d} cores'.format(scaling_script, np.min(CPU_set),np.max(CPU_set)))
        scaling_test_set = CPU_set

    start_time = time.time()

    sim_nx = resolution[0]
    sim_nz = resolution[-1]

    if dim==3:
        if len(resolution) == 3:
            sim_ny = resolution[1]
        else:
            sim_ny = 2*(sim_nx+1)-1
        N_y = []

    N_total_cpu = []
    N_x = []
    N_z = []
    startup_time = []
    wall_time = []
    wall_time_per_iter = []
    work = []
    work_per_core = []

    for CPUs in scaling_test_set:
        if dim == 3:
            res_string = '{:d}x{:d}x{:d}'.format(sim_nx, sim_ny, sim_nz)
            ENV_N_TOTAL_CPU = np.prod(CPUs)
            print(CPUs)
        else:
            res_string = '{:d}x{:d}'.format(sim_nx, sim_nz)
            ENV_N_TOTAL_CPU = CPUs

        print("scaling test of {}".format(scaling_script),
              " at {:s}".format(res_string),
              " on {:d} cores".format(ENV_N_TOTAL_CPU))

        test_env = dict(os.environ,
                        N_X='{:d}'.format(sim_nx),
                        N_Z='{:d}'.format(sim_nz),
                        N_TOTAL_CPU='{:d}'.format(ENV_N_TOTAL_CPU))
        if OpenMPI:
            commands = ["mpirun", "-n","{:d}".format(ENV_N_TOTAL_CPU),
                        "--bind-to", "core", "--map-by", "core"]
        elif MPISGI:
            commands = ['mpiexec_mpt', "-n","{:d}".format(ENV_N_TOTAL_CPU)]
        elif IntelMPI:
             commands = ['mpirun', "-n","{:d}".format(ENV_N_TOTAL_CPU)]
        else:
             commands = ['mpirun', "-n","{:d}".format(ENV_N_TOTAL_CPU)]

        commands += ["python3", scaling_script, "--N_max={:d}".format(sim_nz), "--L_max={:d}".format(sim_nx)]
        if dim == 3:
            commands.append("--mesh={:d},{:d}".format(CPUs[0], CPUs[1]))
            #commands.append("--ny={:d}".format(sim_ny))
            print(" pencils/core (0): {:g}x{:g}={:g}".format(1/2*sim_nx/CPUs[0], sim_ny/CPUs[1], 1/2*sim_nx*sim_ny/(CPUs[0]*CPUs[1])))
            print(" pencils/core (2): {:g}x{:g}={:g}".format(1/2*sim_nx/CPUs[0], 3/2*sim_nz/CPUs[1], 1/2*sim_nx*3/2*sim_nz/(CPUs[0]*CPUs[1])))
            print(" pencils/core (4): {:g}x{:g}={:g}".format(3/2*sim_ny/CPUs[0], 3/2*sim_nz/CPUs[1], 3/2*sim_ny*3/2*sim_nz/(CPUs[0]*CPUs[1])))

        else:
            print(" pencils/core: {:g} ({:g}) and {:g} ({:g})".format(1/2*sim_nx/ENV_N_TOTAL_CPU, 3/2*sim_nx/ENV_N_TOTAL_CPU,
                                                                          sim_nz/ENV_N_TOTAL_CPU, 3/2*sim_nz/ENV_N_TOTAL_CPU))

        if niter is not None:
            commands += ["--run_time_iter={:d}".format(niter)]

        print("command: "+" ".join(commands))
        proc = subprocess.run(commands,
                              env=test_env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = proc.stdout, proc.stderr

        if verbose:
            for line in stdout.splitlines():
                print("out: {}".format(line))

            for line in stderr.splitlines():
                print("err: {}".format(line))

        for line in stdout.splitlines():
            if line.startswith('scaling:'):
                split_line = line.split()
                print(split_line)
                N_total_cpu.append(num(split_line[1]))

                N_x.append(num(split_line[2]))
                N_z.append(num(split_line[3]))

                startup_time.append(num(split_line[4]))
                wall_time.append(num(split_line[5]))
                wall_time_per_iter.append(num(split_line[6]))

                work.append(num(split_line[7]))
                work_per_core.append(num(split_line[8]))

    # change data storage to numpy arrays
    N_total_cpu = np.array(N_total_cpu)
    N_x = np.array(N_x)
    N_z = np.array(N_z)
    startup_time = np.array(startup_time)
    wall_time = np.array(wall_time)
    wall_time_per_iter = np.array(wall_time_per_iter)
    work = np.array(work)
    work_per_core = np.array(work_per_core)

    print(40*'-')
    print("scaling results")
    for i, temp in enumerate(N_total_cpu):
        print(N_total_cpu[i], N_z[i], startup_time[i], wall_time[i], wall_time_per_iter[i])

    data_set = dict()
    data_set['script'] = scaling_script
    data_set['sim_nx'] = sim_nx
    data_set['sim_nz'] = sim_nz
    if dim == 3:
        data_set['sim_ny'] = sim_ny
        data_set['N_y'] = N_x # hack
    data_set['N_total_cpu'] = N_total_cpu
    data_set['N_x'] = N_x
    data_set['N_z'] = N_z
    data_set['startup_time'] = startup_time
    data_set['wall_time'] = wall_time
    data_set['wall_time_per_iter'] = wall_time_per_iter
    data_set['work'] = work
    data_set['work_per_core'] = work_per_core
    data_set['file_label'] = res_string
    data_set['dim'] = dim
    if dim == 3:
        data_set['plot_label'] = r'${:d}\times{:d}\times{:d}$'.format(sim_nx, sim_ny, sim_nz)
        data_set['plot_label_short'] = r'${:d}^3$'.format(sim_nz)
        mesh = [CPUs[0], CPUs[1]]
        data_set['mesh'] = mesh
        data_set['N_x_cpu'] = mesh[0]
        data_set['N_y_cpu'] = mesh[1]
    else:
        data_set['plot_label'] = r'${:d}\times{:d}$'.format(sim_nx, sim_nz)
        data_set['plot_label_short'] = r'${:d}^2$'.format(sim_nz)
        data_set['mesh'] = None

    if not label is None:
        data_set['plot_label'] = data_set['plot_label'] + "-" + label
    write_scaling_run(data_set, label=label)

    end_time = time.time()
    print(40*'*')
    print('time to test {:s}: {:8.3g}'.format(res_string, end_time-start_time))
    print(40*'*')

    return data_set

def write_scaling_run(data_set, label=None):
    file_name = 'scaling_data_'+data_set['file_label']
    if not label is None:
        file_name = file_name+'_'+label
    file_name = file_name+'.db'

    print("writing file {}".format(file_name))
    scaling_file = shelve.open(file_name, flag='n')
    data_set['file_name'] = file_name
    scaling_file['data'] = data_set
    scaling_file.close()

def read_scaling_run(file):
    print("opening file {}".format(file))
    scaling_file = shelve.open(file, flag='r')
    data_set = scaling_file['data']
    scaling_file.close()
    return data_set

# Plotting routines
def plot_scaling_run(data_set, ax_set,
                     ideal_curves = True, scale_to = False, scale_to_resolution=None,
                     linestyle='solid', marker='o', color='None',
                     explicit_label = True, clean_plot=False,
                     dim=None, verbose=False):

    sim_nx = data_set['sim_nx']
    sim_nz = data_set['sim_nz']
    N_total_cpu = data_set['N_total_cpu']
    N_x = data_set['N_x']
    N_z = data_set['N_z']
    if dim is None:
        if 'dim' in data_set:
            dim = data_set['dim']
        else:
            dim = 2
    if dim==3:
        sim_ny = data_set['sim_ny']
        N_y = data_set['N_y']
        N_x_cpu = data_set['N_x_cpu']
        N_y_cpu = data_set['N_y_cpu']

    startup_time = data_set['startup_time']
    wall_time = data_set['wall_time']
    wall_time_per_iter = data_set['wall_time_per_iter']
    work = data_set['work']
    work_per_core = data_set['work_per_core']

    if dim == 2:
        resolution = [sim_nx, sim_nz]
        if scale_to_resolution is None:
            scale_to_resolution = [128,128]
    elif dim == 3 :
        resolution = [sim_nx, sim_ny, sim_nz]
        if scale_to_resolution is None:
            scale_to_resolution = [128,128,128]

    if color is 'None':
        color=next(ax_set[0]._get_lines.prop_cycler)['color']

    scale_to_factor = np.prod(np.array(scale_to_resolution))/np.prod(np.array(resolution))
    scale_factor_inverse = np.int(np.rint((1./scale_to_factor)**(1/dim)))

    if clean_plot:
        plot_label = data_set['plot_label'].split('-')[0]
    else:
        plot_label = data_set['plot_label']

    if explicit_label:
        label_string = plot_label
        scaled_label_string = plot_label + r'$/{:d}^{:d}$'.format(scale_factor_inverse, dim)
    else:
        label_string = data_set['plot_label_short']
        scaled_label_string = data_set['plot_label_short'] + r'$/{:d}^{:d}$'.format(scale_factor_inverse, dim)

    if ideal_curves:
        ideal_cores = np.sort(N_total_cpu)
        i_min = np.argmin(N_total_cpu)
        ideal_time = wall_time[i_min]*(N_total_cpu[i_min]/ideal_cores)
        ideal_time_per_iter = wall_time_per_iter[i_min]*(N_total_cpu[i_min]/ideal_cores)

        ax_set[0].plot(ideal_cores, ideal_time, linestyle='--', color='black')

        ax_set[1].plot(ideal_cores, ideal_time_per_iter, linestyle='--', color='black')



    ax_set[0].plot(N_total_cpu, wall_time, label=label_string,
                   marker=marker, linestyle=linestyle, color=color)

    if dim == 3:
        print("resetting linestyle")
        linestyle='None'
    ax_set[1].plot(N_total_cpu, wall_time_per_iter, label=label_string,
                   marker=marker, linestyle=linestyle, color=color)

    ax_set[2].plot(N_total_cpu, work_per_core/1e-6, label=label_string,
                   marker=marker, linestyle=linestyle, color=color)

    ax_set[3].plot(N_total_cpu, startup_time, label=label_string,
                   marker=marker,  linestyle=linestyle, color=color)

    for i in range(4):
        ax_set[i].set_xscale('log', basex=2)
        ax_set[i].set_yscale('log')
        ax_set[i].margins(x=0.05, y=0.05)

    i_max = N_total_cpu.argmax()
    ax_set[4].plot(N_total_cpu[i_max], work_per_core[i_max]/1e-6, label=label_string,
                     marker=marker,  linestyle=linestyle, color=color)

    if scale_to and scale_to_factor != 1:
        print("scaling by {:f} or (1/{:d})^{:d}".format(scale_to_factor, scale_factor_inverse, dim))
        ax_set[0].plot(N_total_cpu, wall_time*scale_to_factor, marker=marker,
                         label=scaled_label_string, linestyle='--', color=color)

        ax_set[1].plot(N_total_cpu, wall_time_per_iter*scale_to_factor, marker=marker,
                         label=scaled_label_string, linestyle='--',color=color)

    if verbose:
        for N, t in zip(N_total_cpu, wall_time_per_iter):
            print(N, t)


def initialize_plots(num_figs, fontsize=12, color_cycle_length=7):
    import scipy.constants as scpconst
    from cycler import cycler

    fig_set = []
    ax_set = []

    x_size = 3.5 # width of single column in inches
    x_size = 7 # width of single column in inches
    y_size = x_size/scpconst.golden

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    cycle_length = min(color_cycle_length, len(colors))

    for i in range(num_figs):
        fig = plt.figure(figsize=(x_size, y_size))
        ax = fig.add_subplot(1,1,1)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)

        ax.set_prop_cycle(cycler('color', colors[:cycle_length]))

        fig_set.append(fig)
        ax_set.append(ax)
    return fig_set, ax_set


def legend_with_ideal(ax, loc='lower left', fontsize=8):
    handles, labels = ax.get_legend_handles_labels()
    idealArtist = plt.Line2D((0,1),(0,0), color='black', linestyle='--')
    ax.legend([handle for i,handle in enumerate(handles)]+[idealArtist],
              [label for i,label in enumerate(labels)]+['ideal'],
              loc=loc, prop={'size':fontsize})

def add_base10_axis(ax):
    #######################################################
    # from http://stackoverflow.com/questions/31803817/how-to-add-second-x-axis-at-the-bottom-of-the-first-one-in-matplotlib
    ax10 = ax.twiny()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Add some extra space for the second axis at the bottom
    #fig.subplots_adjust(bottom=0.2)

    # Move twinned axis ticks and label from top to bottom
    ax10.xaxis.set_ticks_position("bottom")
    ax10.xaxis.set_label_position("bottom")

    # Offset the twin axis below the host
    ax10.spines["bottom"].set_position(("axes", -0.15))

    # Turn on the frame for the twin axis, but then hide all
    # but the bottom spine
    ax10.set_frame_on(True)
    ax10.patch.set_visible(False)
    for sp in ax10.spines.values():
        sp.set_visible(False)
    ax10.spines["bottom"].set_visible(True)

    tick_locs = ax.xaxis.get_ticklocs()
    ax10.set_xscale('log', basex=2)
    ax10.grid() # suppress gridlines
    #ax10.grid(b=False) # suppress gridlines
    ax10.set_xticks(tick_locs)
    ax10.set_xticklabels(["{:d}".format(int(V)) for V in tick_locs])
    ax10.set_xlim(xlim)
    ax10.set_ylim(ylim)
    return ax10
    #######################################################

def finalize_plots(fig_set, ax_set, script):

    ax_set[0].set_title('Wall time {}'.format(script))
    ax_set[0].set_xlabel('N-core')
    ax_set[0].set_ylabel('total time [s]')
    legend_with_ideal(ax_set[0], loc='lower left')
    fig_set[0].savefig('scaling_time.png')

    ax10 = add_base10_axis(ax_set[1])
    ax_set[1].set_title('Wall time per iteration {}'.format(script))
    ax_set[1].set_xlabel('N-core')
    ax_set[1].set_ylabel('time/iter [s]')
    legend_with_ideal(ax_set[1], loc='lower left')
    xlim = ax_set[1].get_xlim()
    ax_set[1].set_xlim(0.9*xlim[0],1.1*xlim[1])
    fig_set[1].subplots_adjust(bottom=0.2)
    fig_set[1].savefig('scaling_time_per_iter.png', dpi=600)

    ax_set[2].set_title('Normalized work {}'.format(script))
    ax_set[2].set_xlabel('N-core')
    ax_set[2].set_ylabel('N-cores * (time/iter/grid) [$\mu$s]')
    ax_set[2].legend(loc='upper left')
    fig_set[2].savefig('scaling_work.png')

    ax_set[3].set_title('startup time {}'.format(script))
    ax_set[3].set_xlabel('N-core')
    ax_set[3].set_ylabel('startup time [s]')
    ax_set[3].legend(loc='lower right')
    fig_set[3].savefig('scaling_startup.png')

    ax_set[4].set_title('Normalized work {}'.format(script))
    ax_set[4].set_xlabel('N-core')
    ax_set[4].set_ylabel('N-cores * (time/iter/grid) [$\mu$s]')
    ax_set[4].legend(loc='upper left')
    fig_set[4].savefig('scaling_work_strong.png')



if __name__ == "__main__":

    import logging
    logger = logging.getLogger(__name__)

    from docopt import docopt

    fig_set, ax_set = initialize_plots(5)
    args = docopt(__doc__)
    dim = 3

    if args['run']:
        n_L = num(args['--L_max'])
        if args['--N_max'] is not None:
            n_r = num(args['--N_max'])
        else:
            n_r = int((n_L+1)/2)-1
        n_z = n_L+1
        resolution = [n_L,n_r]

        n_z_2 = np.log(n_z)/np.log(2) -1 # 2 pencils per core min
        n_z_2_min = n_z_2-3
        if n_z >= 128:
            n_z_2_min = n_z_2-2
        if args['--one-pencil']:
            print("Pushing to one pencil per core in coeff space; this may be inefficient depending on dealias padding choice.")
            n_z_2 = np.log(n_z)/np.log(2)
        if args['--limit-mem']:
                n_z_2_min += 1
        if args['--max-cores'] is not None:
            log2_max = np.log(np.int(args['--max-cores']))/np.log(2)
            log2_max = log2_max/2
            log2_max = np.floor(log2_max)

            print("max cores in log2 space {}".format(log2_max))
            if n_z_2 > log2_max:
                n_z_2 = log2_max
            max_cores = np.int(args['--max-cores'])
        else:
            max_cores = None

        if args['--min-cores'] is not None:
            log2_min = np.log(np.int(args['--min-cores']))/np.log(2)
            log2_min = log2_min/2
            log2_min = np.ceil(log2_min)

            print("min cores in log2 space {}".format(log2_min))
            if n_z_2_min > log2_min:
                n_z_2_min = log2_min
            min_cores = np.int(args['--min-cores'])
        else:
            min_cores = None

        n_z_2_min = np.ceil(n_z_2_min)
        n_z_2 = np.floor(n_z_2)

        logger.info("Spanning log-2 space from {} -- {}".format(n_z_2_min, n_z_2))
        print("Spanning log-2 space from {} -- {}".format(n_z_2_min, n_z_2))
        CPU_set = (2**np.arange(n_z_2_min, n_z_2+1)).astype(int)[::-1] # flip order so large numbers of cores are done first (and arange goes to -1 of top)
        print("scaling run with {} on {} cores".format(resolution, CPU_set))

        start_time = time.time()
        data_set = do_scaling_run(args['<scaling_script>'], resolution, CPU_set,
                                  niter=int(float(args['--niter'])),
                                  max_cores=max_cores, min_cores=min_cores,
                                  verbose=args['--verbose'], label=args['--label'], dim=dim,
                                  OpenMPI=args['--OpenMPI'], MPISGI=args['--MPISGI'], IntelMPI=args['--IntelMPI'])
        end_time = time.time()

        plot_scaling_run(data_set, ax_set)
        script = args['<scaling_script>']

        print(40*'=')
        print('time to do all tests: {:f}'.format(end_time-start_time))
        print(40*'=')

    elif args['plot']:
        output_path = pathlib.Path(args['--output']).absolute()
        if not output_path.exists():
            output_path.mkdir()
        if not args['--rescale'] is None:
            n_z_rescale = num(args['--rescale'])

            scale_to_resolution = [2*n_z_rescale, n_z_rescale]
            scale_to = True
        else:
            scale_to_resolution = [1, 1]
            scale_to = False

        for file in args['<files>']:
            data_set = read_scaling_run(file)
            plot_scaling_run(data_set, ax_set, scale_to=scale_to, scale_to_resolution=scale_to_resolution, clean_plot=args['--clean_plot'], verbose=args['--verbose'])
        script = data_set['script']

    finalize_plots(fig_set, ax_set, script)
