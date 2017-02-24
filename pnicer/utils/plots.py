# -----------------------------------------------------------------------------
# Import packages
import numpy as np

from itertools import combinations


# -----------------------------------------------------------------------------
def finalize_plot(path=None, dpi=150):
    """
    Helper method to save or show plot.

    Parameters
    ----------
    path : str, optional
        If set, the path where the figure is saved
    dpi : int, optional
        DPI of save figure. Default is 150

    """

    # Import matplotlib
    from matplotlib import pyplot as plt

    # Save or show figure
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close()


# -----------------------------------------------------------------------------
def caxes(ndim, ax_size=None, labels=None):
    """
    Creates a grid of axes to plot all combinations of data.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    ax_size : list, optional
        Single axis size. Default is [3, 3].
    labels : iterable, optional
        Optional list of feature names

    Returns
    -------
    tuple
        tuple containing the figure and a list of the axes.

    """

    # Import
    from matplotlib import pyplot as plt

    if labels is not None:
        if len(labels) != ndim:
            raise ValueError("Number of provided labels must match dimensions")

    if ax_size is None:
        ax_size = [3, 3]

    # Get all combinations to plot
    c = combinations(range(ndim), 2)

    # Create basic plot layout
    fig, axes = plt.subplots(ncols=ndim - 1, nrows=ndim - 1, figsize=[(ndim - 1) * ax_size[0], (ndim - 1) * ax_size[1]])

    if ndim == 2:
        axes = np.array([[axes], ])

    # Adjust plots
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

    axes_out = []
    for idx in c:

        # Get index of subplot
        x_idx, y_idx = ndim - idx[0] - 2, ndim - idx[1] - 1

        # Grab axis
        ax = axes[x_idx, y_idx]

        # Hide tick labels
        if x_idx < ndim - 2:
            ax.axes.xaxis.set_ticklabels([])
        if y_idx > 0:
            ax.axes.yaxis.set_ticklabels([])

        # Add axis labels
        if labels is not None:
            if ax.get_position().x0 < 0.11:
                ax.set_ylabel("$" + labels[idx[0]] + "$")
            if ax.get_position().y0 < 0.11:
                ax.set_xlabel("$" + labels[idx[1]] + "$")

        # Append axes to return list
        axes_out.append(axes[x_idx, y_idx])

        # Delete not necessary axes
        if ((idx[0] > 0) | (idx[1] - 1 > 0)) & (idx[0] != idx[1] - 1):
            fig.delaxes(axes[idx[0], idx[1] - 1])

    return fig, axes_out


# -----------------------------------------------------------------------------
def caxes_delete_ticklabels(axes, xfirst=False, xlast=False, yfirst=False, ylast=False):
    """
    Deletes tick labels from a combination axes list.

    Parameters
    ----------
    axes : iterable
        The combination axes list.
    xfirst : bool, optional
        Whether the first x label should be deleted.
    xlast : bool, optional
        Whether the last x label should be deleted.
    yfirst : bool, optional
        Whether the first y label should be deleted.
    ylast : bool, optional
        Whether the last y label should be deleted.


    """

    # Loop through the axes
    for ax, idx in zip(axes, combinations(range(len(axes)), 2)):

        # Modify x ticks
        if idx[0] == 0:

            # Grab ticks
            xticks = ax.xaxis.get_major_ticks()

            # Conditionally delete
            if xfirst:
                xticks[0].set_visible(False)
            if xlast:
                xticks[-1].set_visible(False)

        if idx[1] == np.max(idx):

            # Grab ticks
            yticks = ax.yaxis.get_major_ticks()

            # Conditionally delete
            if yfirst:
                yticks[0].set_visible(False)
            if ylast:
                yticks[-1].set_visible(False)


# -----------------------------------------------------------------------------
def plot_gmm(gmm, path=None, ax_size=None, draw_components=False, **kwargs):
    """
    Simple plotting routine for GMM instance

    Parameters
    ----------
    gmm : GaussianMixture
        The model to be plotted.
    path : str, optional
        Path to file when the plot should be saved.
    ax_size : list, optional
        List of axis size [xsize, ysize]. Defaults to [7, 5]
    draw_components : bool, optional
        Whether to also draw the components of the GMM. Default is True.
    kwargs
        Any additional pyplot.plot keyword argument (color, lw, etc.)

    """

    # Set default axis size
    if ax_size is None:
        ax_size = [7, 5]

    # Import
    from matplotlib import pyplot as plt
    from pnicer.utils.gmm import gmm_sample_xy, gmm_sample_xy_components

    # Draw samples
    x, y = gmm_sample_xy(gmm=gmm, kappa=4, sampling=20)

    # Create figure and plot data
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=ax_size)
    ax.plot(x, y, **kwargs)

    # Sample and draw components if set
    if draw_components:
        xc, yc = gmm_sample_xy_components(gmm=gmm, kappa=4, sampling=20)

        for y in yc:
            ax.plot(xc, y, color="gray", lw=1, ls="dashed")

    # Save or show
    finalize_plot(path=path)
