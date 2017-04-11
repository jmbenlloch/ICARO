"""
Define plotting functions that return a Figure (rather than using pyplot)
Used by GUI applications
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from invisible_cities.core.mpl_functions import set_plot_labels, circles
from invisible_cities.core.core_functions import define_window
from invisible_cities.core.core_functions import loc_elem_1d
from invisible_cities.core.system_of_units_c import units


def _hist_outline(dataIn, *args, **kwargs):
    (histIn, binsIn) = np.histogram(dataIn, bins='auto', *args, **kwargs)

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return (bins, data)

def hist_1d(data, xlo, xhi):
    """Returns a Figure corresponding to 1d histogram"""
    (bins, n) = _hist_outline(data)
    ylo = 0
    yhi = max(n) * 1.1

    fig = Figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(bins, n, 'k-')
    ax1.axis([xlo, xhi, ylo, yhi])

    return fig


def fplot_vector(v, figsize=(10,10)):
    """Plot vector v and return figure """
    fig = Figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(v)
    return fig



def fplot_pmt_waveforms(pmtwfdf, zoom=False, window_size=800, figsize=(10,10)):
    """plot PMT wf and return figure"""
    fig = Figure(figsize=figsize)
    for i in range(len(pmtwfdf)):
        first, last = 0, len(pmtwfdf[i])
        if zoom:
            first, last = define_window(pmtwfdf[i], window_size)

        ax = fig.add_subplot(3, 4, i+1)
        set_plot_labels(xlabel="samples", ylabel="adc")
        ax.plot(pmtwfdf[i][first:last])
    return fig


def fplot_signal_vs_time_mus(signal,
                            t_min      =    0,
                            t_max      = 1200,
                            signal_min =    0,
                            signal_max =  200,
                            figsize=(10,10)):
    """Plot signal versus time in mus (tmin, tmax in mus). """
    fig = Figure(figsize=figsize)
    tstep = 25 # in ns
    PMTWL = signal.shape[0]
    signal_t = np.arange(0., PMTWL * tstep, tstep)/units.mus
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlim([t_min, t_max])
    ax1.set_ylim([signal_min, signal_max])
    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "signal (pes/adc)")
    ax1.plot(signal_t, signal)
    return fig



def fplot_pmt_signals_vs_time_mus(pmt_signals,
                                 pmt_active,
                                 t_min      =    0,
                                 t_max      = 1200,
                                 signal_min =    0,
                                 signal_max =  200,
                                 figsize=(10,10)):
    """Plot PMT signals versus time in mus  and return figure."""

    tstep = 25
    PMTWL = pmt_signals[0].shape[0]
    signal_t = np.arange(0., PMTWL * tstep, tstep)/units.mus
    fig = Figure(figsize=figsize)

    for j, i in enumerate(pmt_active):
        ax1 = fig.add_subplot(3, 4, j+1)
        ax1.set_xlim([t_min, t_max])
        ax1.set_ylim([signal_min, signal_max])
        set_plot_labels(xlabel = "t (mus)",
                        ylabel = "signal (pes/adc)")

        ax1.plot(signal_t, pmt_signals[i])

    return fig


def fplot_s12(S12, figsize=(10,10)):
    """Plot the peaks of input S12 and return figure..

    S12 is a dictionary
    S12[i] for i in keys() are the S12 peaks
    """
    fig = Figure(figsize=figsize)
    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "S12 (pes)")
    xy = len(S12)
    if xy == 1:
        t = S12[0][0]
        E = S12[0][1]
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(t/units.mus, E)
    else:
        x = 3
        y = xy/x
        if y % xy != 0:
            y = int(xy/x) + 1
        for i in S12.keys():
            ax1 = fig.add_subplot(x, y, i+1)
            t = S12[i][0]
            E = S12[i][1]
            ax1.plot(t/units.mus, E)
    return fig


def fplot_s2si_map(S2Si, xs, ys, cmap='Blues', figsize=(10,10)):
        """Plot a map of the energies of S2Si objects."""
        radius = 2
        fig = Figure(figsize=figsize)
        r = np.ones(len(xs)) * radius
        col = np.zeros(len(xs))
        for sipm in S2Si.values():
            for nsipm, E in sipm.items():
                ene = np.sum(E)
                col[nsipm] = ene
        ax1 = fig.add_subplot(aspect="equal")
        circles(xs, ys, r, c=col, alpha=0.5, ec="none", cmap=cmap)
        ax1.colorbar()

        ax1.xlim(-198, 198)
        ax1.ylim(-198, 198)

        return fig

def fill_col_vector(S2Si, dim):

    col = np.zeros(dim)
    for sipm in S2Si.values():
        for nsipm, E in sipm.items():
            ene = np.sum(E)
            col[nsipm] = ene
    return col

def add_simap(S2Si, xs, COL):

    col = fill_col_vector(S2Si, len(xs))
    COL.append(col)

def get_simap(x, y, col, radius=5, cmap='Blues', figsize=(10,10)):

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    r = np.ones(len(x)) * radius

    im = ax.scatter(x, y, c=col, s=r, cmap=cmap)
    fig.colorbar(im, ax=ax)
    im.set_clim(0.0, 1.0)
    return fig

def fplot_map(COL, xs, ys, radius=5, cmap='Blues', figsize=(10,10)):

    col = COL[0]
    for c in COL[1:]:
        col+=c
    return get_simap(xs, ys, col, radius, cmap=cmap, figsize=figsize)

def fplot_s2si_map(S2Si, xs, ys, radius=5,cmap='Blues', figsize=(10,10)):

    col = fill_col_vector(S2Si, len(xs))
    c = col/max(col)
    return get_simap(xs, ys, c, radius, cmap=cmap, figsize=figsize)
