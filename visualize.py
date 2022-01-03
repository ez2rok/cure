import plotly

import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd

from icecream import ic
from matplotlib import animation

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


def matplotlib_animation(embedding_history, y, labels=None, n_bins=20):

    # initialize figure
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.style.use('ggplot')

    limits = get_axis_limits(embedding_history, y, n_bins)
    plt.style.use('fivethirtyeight')

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
    anim.save('cure.mp4', writer='ffmpeg')
    return anim

# def plot_embedding(embedding, n_bins, colors, ax, target_names):
#   """
#   plot the embedded data as a histogram to show the uncoupling between the two clusters
#   """

#   # initial values
#   offset = 50
#   n_clusters = len(target_names)

#   # seperate the datapoints that belong to different classes
#   seperated_data = [embedding[i * offset : (i+1) * offset] for i in range(n_clusters)]

#   # plot the mapped data
#   heights, bins, patches = plt.hist(seperated_data,
#                                     bins=n_bins,
#                                     alpha=0.75,
#                                     stacked=True,
#                                     color=colors)

#   # plot legend
#   handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in colors]
#   plt.legend(handles, target_names)

#   # make vertical line at x = 0
#   ax.axvline(linestyle='--', color='red')

#   # title the graph
#   ax.set_title(' Uncoupled Distributions via CURE')
