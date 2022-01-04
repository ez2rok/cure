import plotly

import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd

from icecream import ic
from matplotlib import animation


def plot_data(X, y, title=None, labels=None, filename='data', outdir='./figures', save=False):
    """
    Plot 2D data. Each class is plotted in a different color.

    Parameters
    ----------
    X : (n_samples, n_features) array
        The data.
    y : (n_samples,) array
        The labels for the data.
    title : str, optional
        The title of the figure, by default None.
    labels : list of str, optional
        The labels for the classes, by default None.
    filename : str, optional
        The name of the file that we are saving this figure in, by default 'data'.
    outdir : str, optional
        The name of the directory that we are saving this figure in, by default './figures/'.
    save : bool, optional
        If true, save this figure, by default False.

    Returns
    -------
    matplotlib.figure.Figure
        A figure that plots each class of the data in a different color.
    """

    # plot the data
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, y_val in enumerate(np.unique(y)):
        label = labels[i] if labels is not None else 'Class {}'.format(y_val)
        ax.scatter(X[y == y_val, 0], X[y == y_val, 1], label=label, s=10)

    # format the data
    limit = np.max(np.abs(X)) * 1.03
    ax.set(xlabel='x', ylabel='y', title=title,
           xlim=(-limit, limit), ylim=(-limit, limit))
    plt.legend()

    # save the data
    if save:
        fig.savefig(outdir + '/' + filename + '.png')
    return fig


###############################################################################
# Plotly Animation
###############################################################################


def plotly_animation(embedding_history, y, n_bins=20):

    # initial values
    n_iterations = embedding_history.shape[0]
    n_samples = embedding_history.shape[1]

    # make dataframe
    data = {'Data Embedding': embedding_history.flatten(),
            'iteration': np.repeat(np.arange(n_samples), n_iterations),
            'label': np.tile(y, n_iterations)}
    df = pd.DataFrame(data)

    # make animation
    fig = px.histogram(df, x='Data Embedding', color='label', n_bins=n_bins,
                       range_x=[np.min(embedding_history),
                                np.max(embedding_history)],
                       animation_frame='iteration', title='CURE Data Embedding')
    fig.add_vline(x=0, line_width=3, line_dash="dash")

    # change speed of animation in milliseconds
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 1

    # save animation
    plotly.offline.plot(fig, filename='figures/fig1.html')


###############################################################################
# Matplotlib Animation
###############################################################################


def get_axis_limits(embedding_history, y, n_bins):
    """
    Get the lower and upper limits of the x, y axes.

    Parameters
    ----------
    embedding_history : (n_iterations, n_samples) array
        The data embedding at each iteration. Each row is the data embedded with a different set of weights.
    y : (n_samples,) array
        The labels for the data.
    n_bins : int, optional
        Number of bins in the histogram.

    Returns
    -------
    limits : (4,) array
        The lower and upper limits of the x, y axes ordered as array([y_lower, y_upper, x_lower, x_upper]).
    """

    # compute limit of y axis
    heights = []
    for embedding in embedding_history:
        data = [embedding[y == y_val] for y_val in np.unique(y)]
        heights.append(plt.hist(data, bins=n_bins, stacked=True)[0])
    y_upper = np.max(heights)
    y_lower = 0

    # compute limits of x axis
    x_lower = np.min(embedding_history)
    x_upper = np.max(embedding_history)

    limits = np.array([y_lower, y_upper, x_lower, x_upper])
    return limits


def update_figure(frame_number, embedding_history, y, ax, n_bins, limits, labels):
    """
    Update the figure at each frame.

    Parameters
    ----------
    frame_number : int
        The current frame number.
    embedding_history : (n_samples, n_features) array
        The data embedding at each iteration. Each row is the data embedded with a different set of weights.
    y : (n_samples,) array
        The labels for the data.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes that we are plotting the data on.
    n_bins : int, optional
        Number of bins in the histogram.
    limits : (4,) array
        The lower and upper limits of the x, y axes ordered as array([y_lower, y_upper, x_lower, x_upper]).
    labels : list of str, optional
        The unique class name that each (numeric) label corresponds to, by default None.
    """

    # clear axis
    ax.clear()

    # plot new histogram
    embedding = embedding_history[frame_number]
    data = [embedding[y == y_val] for y_val in np.unique(y)]
    ax.hist(data, bins=n_bins, stacked=True, label=labels)

    # format new histogram
    padding = 1.1
    y_lower, y_upper, x_lower, x_upper = limits * padding
    ax.set(xlim=(x_lower, x_upper),
           ylim=(y_lower, y_upper),
           title='CURE Data Embedding',
           xlabel='Data Embedding',
           ylabel='Count')
    ax.axvline(x=0, linestyle='--', color='black')
    ax.legend()


def matplotlib_animation(embedding_history, y, outdir='./figures', filename='cure', labels=None, n_bins=20):
    """
    Create an animation of the data embedding changing over time as the weights change in the CURE algorithm.
    Use matplotlib.

    Parameters
    ----------
    embedding_history : (n_iterations, n_samples) array
        The data embedding at each iteration. Each row is the data embedded with a different set of weights.
    y : (n_samples,) array
        The labels for the data.
    outdir : str, optional
        The animation in saved in a directory with this name, by default './figures'.
    filename : str, optional
        The animation in saved in a file with this name, by default 'cure'.
    labels : list of str, optional
        The unique class name that each (numeric) label corresponds to, by default None.
    n_bins : int, optional
        Number of bins in the histogram, by default 20.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation of the data embedding changing over time as the weights change.
    """

    # initialize figure
    fig, ax = plt.subplots(figsize=(10, 10))
    limits = get_axis_limits(embedding_history, y, n_bins)

    # make and save animation
    anim = animation.FuncAnimation(fig,
                                   update_figure,
                                   repeat=True,
                                   fargs=(embedding_history, y,
                                          ax, n_bins, limits, labels),
                                   frames=len(embedding_history)-1,
                                   interval=500,
                                   repeat_delay=2000
                                   )
    anim.save(outdir + '/' + filename + '.mp4', writer='ffmpeg')
    return anim
