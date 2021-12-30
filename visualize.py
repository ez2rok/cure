import matplotlib.pyplot as plt

def plot_embedded_data(X_embd, target_names):
    plt.hist(X_embd, bins=50, alpha=0.75, stacked=True)



def plot_embedding(embedding, n_bins, colors, ax, target_names):
  """
  plot the embedded data as a histogram to show the uncoupling between the two clusters
  """

  # initial values
  offset = 50
  n_clusters = len(target_names)

  # seperate the datapoints that belong to different classes
  seperated_data = [embedding[i * offset : (i+1) * offset] for i in range(n_clusters)]

  # plot the mapped data
  heights, bins, patches = plt.hist(seperated_data, 
                                    bins=n_bins, 
                                    alpha=0.75,
                                    stacked=True, 
                                    color=colors)

  # plot legend
  handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in colors]
  plt.legend(handles, target_names)

  # make vertical line at x = 0
  ax.axvline(linestyle='--', color='red')

  # title the graph
  ax.set_title(' Uncoupled Distributions via CURE')