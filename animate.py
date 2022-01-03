from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def init():
  n_clusters = 3
  colors = colors = [plt.cm.tab10(x) for x in np.linspace(0, 1, n_clusters)]
  target_names = ['cluster1', 'cluster2', 'cluster3']
  scat = plot_iris_dataset(data, colors, ax, target_names)
  line.set_data([], [])
  return line,

def animate(i, embedding_hist):
  x = np.linspace(0, 4, 1000)
  y = np.sin(2 * np.pi * (x - 0.01 * i))
  line.set_data(x, y)

  offset = 50
  n_bins = 40
  n_clusters = len(target_names)
  colors = [plt.cm.tab10(x) for x in np.linspace(0, 1, n_clusters)]
  embedding = embedding_hist[i]
  seperated_data = [embedding[i * offset : (i+1) * offset] for i in range(n_clusters)]
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
  ax.set_title(' Uncoupled Distributions via CURE')

  return patches

if __name__ == '__main__':

    #load data
    from make_dataset import iris_data
    from sklearn.model_selection import train_test_split as tts
    from loss import get_embedding
    seed = 420
    classes = [2, 1]
    X, y = iris_data(classes)
    X_train, X_test, y_train, y_test = tts(X, y, random_state=seed)

    # run CURE
    from cure import CURE
    cure = CURE()
    weight_history = cure.fit(X_train)[1]
    embedding_history = [get_embedding(X_train, weights) for weights in weight_history]

    fig = plt.figure()
    ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
    line, = ax.plot([], [], lw=3)

    anim = FuncAnimation(fig, 
                        animate, 
                        init_func=init,
                        frames=len(embedding_history)-1,
                        fargs=(embedding_history,),
                        interval=200, 
                        blit=True)