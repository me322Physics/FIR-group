import matplotlib as mpl
from matplotlib import pyplot as plt


def rc_def():
    mpl.rc('lines', linewidth=1.75, markersize=8.0, markeredgewidth=0.75)
    #mpl.rc('font', size=20.0, family="DejaVu Serif")
    mpl.rc('xtick', labelsize='medium')
    mpl.rc('ytick', labelsize='medium')
    mpl.rc('xtick.major', width=1.0, size=8)
    mpl.rc('ytick.major', width=1.0, size=8)
    mpl.rc('xtick.minor', width=1.0, size=4)
    mpl.rc('ytick.minor', width=1.0, size=4)
    mpl.rc('axes', linewidth=1.5)
    mpl.rc('legend', fontsize='small', numpoints=1, labelspacing=0.4, frameon=False) 
    #mpl.rc('text', usetex=True) 
    mpl.rc('savefig', dpi=300)
    return


def make_fig_multi(nrows, ncols, setticks):
    f, ax = plt.subplots(nrows=nrows, ncols=ncols)
    # plt.minorticks_on()
    if setticks:
        ylocator6 = plt.MaxNLocator(5)
        xlocator6 = plt.MaxNLocator(6)
        if len(ax.shape) > 1:
            for axrow in ax:
                for axcol in axrow:
                    axcol.xaxis.set_major_locator(xlocator6)
                    axcol.yaxis.set_major_locator(ylocator6)
        else:
            for axcol in ax:
                axcol.xaxis.set_major_locator(xlocator6)
                axcol.yaxis.set_major_locator(ylocator6)
    return f, ax


def make_fig():
    f = plt.figure()
    ax = plt.subplot(111)
    plt.minorticks_on()
    #ylocator6 = plt.MaxNLocator(5)
    #xlocator6 = plt.MaxNLocator(6)
    #ax.xaxis.set_major_locator(xlocator6)
    #ax.yaxis.set_major_locator(ylocator6)
    return f, ax
