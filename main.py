# experiment1
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.model_selection import train_test_split as tts
from icecream import ic

# import local files
from src.data.make_dataset import iris_data, elliptical_data, add_intercept, fashion_mnist_data
from src.models.cure import CURE
from src.models.evaluate import adjusted_rand, misclassification_rate
from src.models.loss import get_embedding
from src.visualization.visualize import matplotlib_animation, plot_data, plotly_animation


def get_subset(X, y, subset_size1, subset_size2):
    """
    Exctract subset_size1 datapoints from class1 and subset_size2 datapoints 
    from class2 of X and y.
    """

    idxs1 = np.argwhere(y == -1)[:subset_size1]
    idxs2 = np.argwhere(y == 1)[:subset_size2]
    X = np.concatenate((X[idxs1], X[idxs2])).squeeze()
    y = np.concatenate((y[idxs1], y[idxs2])).squeeze()
    return X, y


def experiment1(save=False):
    """
    Experiment1: Run CURE on an elliptically distributed dataset. Then.
        1. Animate how CURE embeds the data.
        2. Plot the true clustering and CURE's predicted clustering.
        3. Evaluate the predicted clustering.

    Parameters
    ----------
    save : bool, optional
        If true, save the figures, by default False

    Returns
    -------
    results : dict
        A dictionary containing the results of the experiment.
    """

    # initial values
    outdir = './reports/figures/experiment1/'
    seed = 420

    # get data
    n = 1000
    d = 2
    labels = ['Class 1', 'Class 2']
    X, y = elliptical_data(n, d, seed=seed, mu_val=[0, 4], sigma_val=5)

    # run cure
    cure = CURE(random_state=seed)
    cure.fit(add_intercept(X), record_history=True)
    weight_history = cure.weight_history
    y_pred = cure.predict(add_intercept(X))
    embedding_history = get_embedding(weight_history, add_intercept(X))

    # animate cure
    file = outdir + 'cure_animation.html' if save else None
    anim_plotly = plotly_animation(embedding_history, y, file, labels=labels)

    file = outdir + 'cure_animation.mp4' if save else None
    anim_matplt = matplotlib_animation(
        embedding_history, y, file, labels=labels)

    # plot data
    file = outdir + 'true_clustering_fig.png' if save else None
    true_clustering_fig = plot_data(X, y, file=file, title='True Clustering',
                                    labels=labels)
    file = outdir + 'cure_clustering_fig.png' if save else None
    cure_clustering_fig = plot_data(X, y_pred, file, title='Predicted Clustering via CURE',
                                    labels=labels)

    # evaluate predictions
    adj_rand = adjusted_rand(y, y_pred)
    misclf = misclassification_rate(y, y_pred)
    print('Adjusted Rand Index = {:.3f}\nMisclassification Rate = {:.3f}%'.format(
        adj_rand, misclf * 100))

    # assert statements to verify everything is working
    adj_rand_true = 0.9880240131963988
    misclf_true = 0.0030000000000000027
    np.testing.assert_allclose(
        adj_rand, adj_rand_true, err_msg='Adjusted Rand Index is incorrect. You done goofed!')
    np.testing.assert_allclose(
        misclf, misclf_true, err_msg='Misclassification Rate is incorrect. You done goofed!')

    results = {'anim_plotly': anim_plotly, 'anim_matplt': anim_matplt,
               'true_clustering_fig': true_clustering_fig, 'cure_clustering_fig': cure_clustering_fig,
               'adj_rand': adj_rand, 'misclf': misclf}

    return results


def experiment2(save=False):
    """
    Compare the performance of CURE and many other clustering algorithms on various datasets.
    The clustering algorithms are: 
        CURE
        KMeans
        Meanshift
        Spectral clustering
        Ward
        Agglomerative Clustering
        DBSCAN
        OPTICS
        BIRCH
        Gaussian Mixture

    Much of this code is taken form the sklearn tutorial here.
    https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py

    Parameters
    ----------
    save : bool, optional
        If true, save the figure, by default False.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        The different clusterings and their performance.
    """

    np.random.seed(0)
    file = './reports/experiment2/compare_clustering.png'

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 1500

    noisy_circles = datasets.make_circles(n_samples=n_samples,
                                          factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    blobs = datasets.make_blobs(n_samples=n_samples, centers=2, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(
        n_samples=n_samples, centers=2, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5],
                                 random_state=random_state, centers=2)

    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(9 * 2 + 3, 13))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
    )

    plot_num = 1

    default_base = {
        "quantile": 0.3,
        "eps": 0.3,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 10,
        "n_clusters": 2,
        "min_samples": 20,
        "xi": 0.05,
        "min_cluster_size": 0.1,
    }

    datasets_ = [
        (
            noisy_circles,
            {
                "damping": 0.77,
                "preference": -240,
                "quantile": 0.2,
                "n_clusters": 2,
                "min_samples": 20,
                "xi": 0.25,
            },
        ),
        (noisy_moons, {"damping": 0.75, "preference": -220, "n_clusters": 2}),
        (
            varied,
            {
                "eps": 0.18,
                "n_neighbors": 2,
                "min_samples": 5,
                "xi": 0.035,
                "min_cluster_size": 0.2,
            },
        ),
        (
            aniso,
            {
                "eps": 0.15,
                "n_neighbors": 2,
                "min_samples": 20,
                "xi": 0.1,
                "min_cluster_size": 0.2,
            },
        ),
        (blobs, {}),
        (no_structure, {})
    ]

    for i_dataset, (dataset, algo_params) in enumerate(datasets_):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params["n_neighbors"], include_self=False
        )
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # ============
        # Create cluster objects
        # ============
        cure = CURE()
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"])
        ward = cluster.AgglomerativeClustering(
            n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
        )
        spectral = cluster.SpectralClustering(
            n_clusters=params["n_clusters"],
            eigen_solver="arpack",
            affinity="nearest_neighbors",
        )
        dbscan = cluster.DBSCAN(eps=params["eps"])
        optics = cluster.OPTICS(
            min_samples=params["min_samples"],
            xi=params["xi"],
            min_cluster_size=params["min_cluster_size"],
        )
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average",
            affinity="cityblock",
            n_clusters=params["n_clusters"],
            connectivity=connectivity,
        )
        birch = cluster.Birch(n_clusters=params["n_clusters"])
        gmm = mixture.GaussianMixture(
            n_components=params["n_clusters"], covariance_type="full"
        )

        clustering_algorithms = (
            ('CURE', cure),
            ("MiniBatch\nKMeans", two_means),
            ("MeanShift", ms),
            ("Spectral\nClustering", spectral),
            ("Ward", ward),
            ("Agglomerative\nClustering", average_linkage),
            ("DBSCAN", dbscan),
            ("OPTICS", optics),
            ("BIRCH", birch),
            ("Gaussian\nMixture", gmm),
        )

        for name, algorithm in clustering_algorithms:
            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the "
                    + "connectivity matrix is [0-9]{1,2}"
                    + " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding"
                    + " may not work as expected.",
                    category=UserWarning,
                )
                algorithm.fit(add_intercept(
                    X), y) if name == 'CURE' else algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(add_intercept(
                    X)) if name == 'CURE' else algorithm.predict(X)

            plt.subplot(len(datasets_), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(
                list(
                    islice(
                        cycle(
                            [
                                "#377eb8",
                                "#ff7f00",
                                "#4daf4a",
                                "#f781bf",
                                "#a65628",
                                "#984ea3",
                                "#999999",
                                "#e41a1c",
                                "#dede00",
                            ]
                        ),
                        int(max(y_pred) + 1),
                    )
                )
            )
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[np.where(
                y_pred == -1, 0, y_pred).astype(int)])
            name = algorithm.__class__.__name__

            text = ("%.2fs" % (t1 - t0)).lstrip("0")
            text += '' if y is None else "\nARI={:.3f}".format(
                adjusted_rand(y, y_pred)).lstrip("0")

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plt.text(
                0.99,
                0.01,
                text,
                transform=plt.gca().transAxes,
                size=15,
                horizontalalignment="right",
            )
            plot_num += 1

        if save:
            plt.savefig(file)
    fig = plt.gcf()
    return fig


def experiment3(save=False):

    def get_subset(X, y, subset_size1, subset_size2):
        """
        Exctract subset_size1 datapoints from class1 and subset_size2 datapoints 
        from class2 of X and y.
        """

        idxs1 = np.argwhere(y == -1)[:subset_size1]
        idxs2 = np.argwhere(y == 1)[:subset_size2]
        X = np.concatenate((X[idxs1], X[idxs2])).squeeze()
        y = np.concatenate((y[idxs1], y[idxs2])).squeeze()
        return X, y

    def get_results(seed, n, d, metric):
        X, y = elliptical_data(
            n, d, seed, mu_val=[0, 4], sigma_val=5)
        X = add_intercept(X)

        subset_size1 = 600
        subset_size2s = [600 // i for i in range(1, 5)]
        clfs = [cluster.KMeans(n_clusters=2, random_state=seed),
                CURE(random_state=seed)]

        results = []

        for clf in clfs:
            result = []
            for subset_size2 in subset_size2s:

                X_sub, y_sub = get_subset(X, y, subset_size1, subset_size2)
                X_train, X_test, y_train, y_test = tts(
                    X_sub, y_sub, random_state=seed)

                clf.fit(X_train)
                y_pred = clf.predict(X_test)
                score = metric(y_test, y_pred)
                result.append(score)
            results.append(result)

        return results

    # initial values
    n = 1300
    d = 2
    n_trials = 5
    seeds = [i for i in range(n_trials)]

    # Adjusted Rand Index (ARI)
    ari_results = np.array(
        [get_results(seed, n, d, adjusted_rand) for seed in seeds])
    ari_means = np.mean(ari_results, axis=0)
    ari_stds = np.std(ari_results, axis=0)

    ari_text = [['{:.3f} ?? {:.3f}'.format(m, s).lstrip('0') for m, s in zip(m_s, s_s)]
                for m_s, s_s in zip(ari_means, ari_stds)]

    ratios = ['{} : 1'.format(i) for i in range(1, 5)]
    clfs = ['KMeans ARI', 'CURE ARI']
    df_ari = pd.DataFrame(ari_text, index=clfs, columns=ratios)

    # misclassification rate
    mclf_results = np.array(
        [get_results(seed, n, d, misclassification_rate) for seed in seeds])
    mclf_means = np.mean(mclf_results, axis=0)
    mclf_stds = np.std(mclf_results, axis=0)

    mclf_text = [['{:.1f} ?? {:.1f}%'.format(m * 100, s * 100) for m, s in zip(m_s, s_s)]
                 for m_s, s_s in zip(mclf_means, mclf_stds)]

    ratios = ['{} : 1'.format(i) for i in range(1, 5)]
    clfs = ['KMeans MisClf', 'CURE MisClf']
    df_mclf = pd.DataFrame(mclf_text, index=clfs, columns=ratios)

    X, y = elliptical_data(n, d, seeds[0], mu_val=[0, 4], sigma_val=5)
    return df_ari, df_mclf, X, y


def experiment4(save=False):
    """
    Compare the adjusted rand index (ARI) and the misclassification rate of CURE against those of
    K-Means, Spectral Clustering (vanilla), and Spectral Clustering (Gaussian kernel). 
    Run these classifiers on the T-shirt and hoodie classes from the Fashion MNIST dataset.

    Parameters
    ----------
    save : bool, optional
        If true, save the figures, by default False

    Returns
    -------
    dfs : list of DataFrames
        The first element of the list is the dataframe containing the ARI scores for the four
        clustering algorithms. The second element is the dataframe containing the misclassification
        rates for the four clustering algorithms.
    """

    # initial values
    file = './reports/experiment4/fashion_mnist.png'
    seed = 420

    # class ratios
    # n1s = [6000] * 4
    # n2s = [6000, 3000, 2000, 1500]
    # m1s = [1000] * 4
    # m2s = [1000, 500, 333, 250]
    n1s = [60, 60]  # number of training datapoints in class 1
    n2s = [2, 4]  # number of training datapoints in class 2
    m1s = [30, 30]  # number of testing datapoints in class 1
    m2s = [1, 2]  # number of testing datapoints in class 2
    n_class1s = [m1 + n1 for m1, n1 in zip(m1s, n1s)]
    n_class2s = [m2 + n2 for m2, n2 in zip(m2s, n2s)]

    # initialize classifiers
    n_inits = 3  # number of times to run and initialize the CURE classifier
    clfs = [CURE(random_state=seed),
            cluster.KMeans(n_clusters=2, random_state=seed),
            cluster.SpectralClustering(
                n_clusters=2, random_state=seed, affinity='nearest_neighbors'),
            cluster.SpectralClustering(n_clusters=2, random_state=seed)]

    # initialize result arrays
    misclf_results = np.empty((len(clfs), len(n1s)), dtype=object)
    ari_results = np.empty((len(clfs), len(n1s)), dtype=object)

    # plotting values
    clf_names = ['CURE', 'K-Means',
                 'Spectral Clustering\n(vanilla)', 'Spectral Clustering\n(Gaussian kernel)']
    class_ratios = ['{}:1'.format(n1 // n2)
                    for (n1, n2) in zip(n1s, n2s)]

    # set up plot
    fig, axs = plt.subplots(len(n1s), len(clfs) + 1,
                            figsize=(5 * len(clfs), 10), sharey=True, sharex=True)
    [ax.set(adjustable='box', aspect='equal') for ax in axs.flatten()]
    fig.suptitle(
        'CURE and Other Clustering Algorithms on Fashion MNIST (PCA): ', size=24)
    plt.setp(fig.get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    fig.subplots_adjust(left=0.02, wspace=0.05)

    # loop through different class ratios of the data
    for i, (n_class1, n_class2, m1, m2) in enumerate(zip(n_class1s, n_class2s, m1s, m2s)):

        # get data
        X, y = fashion_mnist_data(n_class1=n_class1, n_class2=n_class2)
        X_train, y_train = X[m1:], y[m1:]

        # plot data
        ax = axs[i, 0]
        plot_data(X, y, ax=ax, labels=['T-Shirt', 'Pullover'])
        ax.set_ylabel(class_ratios[i], fontsize=16)
        ax.set_xlabel(None)
        if i == 0:
            axs[i, 0].set_title('True Clustering', size=18)

        # plot text
        text = 'ARI: 100.0%\nMisClf: 0.0%'
        ax.text(0.99, 0.01, text, transform=ax.transAxes,
                size=13, horizontalalignment="right")

        # loop through different classifiers
        for j, clf in enumerate(clfs):

            # catch warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the "
                    + "connectivity matrix is [0-9]{1,2}"
                    + " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding"
                    + " may not work as expected.",
                    category=UserWarning,
                )

                # run classification algorithm and make predictions
                if type(clf).__name__ == 'CURE':
                    y_preds = []
                    for _ in range(n_inits):
                        clf.fit(add_intercept(X_train), y_train)
                        y_pred = clf.predict(add_intercept(X))
                        y_preds.append(y_pred)
                        clf = CURE(random_state=seed + 1)
                elif type(clf).__name__ == 'KMeans':
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X)
                else:  # Spectral Clustering
                    y_pred = clf.fit_predict(X)

            # evaluate and record predictions
            if type(clf).__name__ == 'CURE':
                aris = [adjusted_rand(y, y_pred) for y_pred in y_preds]
                misclfs = [misclassification_rate(
                    y, y_pred) for y_pred in y_preds]
                ari_results[j, i] = '{:.1f}% ?? {:.3f}'.format(
                    np.mean(aris) * 100, np.std(aris))
                misclf_results[j, i] = '{:.1f}% ?? {:.3f}'.format(
                    np.mean(misclfs) * 100, np.std(misclfs))
            else:
                ari_results[j, i] = '{:.1f}%'.format(
                    adjusted_rand(y, y_pred) * 100)
                misclf_results[j, i] = '{:.1f}%'.format(
                    misclassification_rate(y, y_pred) * 100)

            # plot predictions
            ax = axs[i, j + 1]
            plot_data(X, y_pred, ax=ax)
            ax.set(xlabel=None, ylabel=None)
            if i == 0:
                ax.set_title(clf_names[j], size=18)

            # plot text
            text = 'ARI: {}\nMisClf: {}'.format(
                ari_results[j, i], misclf_results[j, i])
            ax.text(0.99, 0.01, text, transform=ax.transAxes,
                    size=13, horizontalalignment="right")

            # save figure
            if save:
                fig.savefig(file)

    # write results to file
    dfs = []
    dir = './reports/experiment4/'
    filenames = [dir + 'adjusted_rand_index.csv',
                 dir + 'misclassification.csv']
    for filename, result in zip(filenames, [ari_results, misclf_results]):
        df = pd.DataFrame(result, columns=class_ratios, index=clf_names)
        df.columns.name = 'Class Ratio'
        if save:
            df.to_csv(filename)
        dfs.append(df)
    ari_df, misclf_df = tuple(dfs)
    return fig, ari_df, misclf_df


if __name__ == '__main__':
    # experiment1(save=True)
    # experiment2(save=True)
    # experiment3()
    fig, ari_df, misclf_df = experiment4(save=True)
    ic(ari_df)
