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
from src.data.make_dataset import iris_data, elliptical_data, add_intercept
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


def experiment1():

    # get data
    seed = 42
    n = 1000
    d = 2
    X, y = elliptical_data(n, d, seed=42, mu_val=[0, 4], sigma_val=5)
    X = add_intercept(X)

    # run cure
    cure = CURE(random_state=seed)
    y_pred = cure.fit_predict(X)

    # plot it
    true_clustering = plot_data(X, y, title='True Clustering')
    predicted_clustering = plot_data(
        X, y, title='Predicted Clustering via CURE')

    # evaluate predictions
    adj_rand = adjusted_rand(y, y_pred)
    misclf = misclassification_rate(y, y_pred)
    print('Adjusted Rand Index = {:.3f}\nMisclassification Rate = {:.3f}%'.format(
        adj_rand, misclf * 100))

    # assert satements to verify everything is working
    adj_rand_true = 0.9840480393607277
    misclf_true = 0.00400000000000000
    np.testing.assert_allclose(
        adj_rand, adj_rand_true, err_msg='Adjusted Rand Index is incorrect. You done goofed!')
    np.testing.assert_allclose(
        misclf, misclf_true, err_msg='Misclassification Rate is incorrect. You done goofed!')


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
    [type]
        [description]
    """

    np.random.seed(0)
    outdir = './figures/'

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
                algorithm.fit(add_intercept(X), y) if name == 'CURE' else algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(add_intercept(X)) if name == 'CURE' else algorithm.predict(X)

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
            plt.savefig(outdir + 'compare_clustering.png')
    fig = plt.gcf()
    return fig


def experiment3():

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

    ari_text = [['{:.3f} ± {:.3f}'.format(m, s).lstrip('0') for m, s in zip(m_s, s_s)]
                for m_s, s_s in zip(ari_means, ari_stds)]

    ratios = ['{} : 1'.format(i) for i in range(1, 5)]
    clfs = ['KMeans ARI', 'CURE ARI']
    df_ari = pd.DataFrame(ari_text, index=clfs, columns=ratios)

    # misclassification rate
    mclf_results = np.array(
        [get_results(seed, n, d, misclassification_rate) for seed in seeds])
    mclf_means = np.mean(mclf_results, axis=0)
    mclf_stds = np.std(mclf_results, axis=0)

    mclf_text = [['{:.1f} ± {:.1f}%'.format(m * 100, s * 100) for m, s in zip(m_s, s_s)]
                 for m_s, s_s in zip(mclf_means, mclf_stds)]

    ratios = ['{} : 1'.format(i) for i in range(1, 5)]
    clfs = ['KMeans MisClf', 'CURE MisClf']
    df_mclf = pd.DataFrame(mclf_text, index=clfs, columns=ratios)

    X, y = elliptical_data(n, d, seeds[0], mu_val=[0, 4], sigma_val=5)
    return df_ari, df_mclf, X, y


def experiment4():

    # get data
    seed = 420
    classes = [2, 1]
    X, y = iris_data(classes)
    X = add_intercept(X)
    X_train, X_test, y_train, y_test = tts(X, y, random_state=seed)

    # run cure
    cure = CURE(random_state=seed)
    weight_history = cure.fit(X_train, record_history=True)[-1]
    embedding_history = get_embedding(weight_history, X_train)

    # animate cure
    file = './figures/cure_animation.html'
    plotly_animation(embedding_history, y_train, file, labels=['Flower 1', 'Flower 2'])
    
    file = './figures/cure_animation.mp4'
    matplotlib_animation(embedding_history, y_train, labels=['Flower 1', 'Flower 2'])


if __name__ == '__main__':
    # experiment1()
    # experiment2(save=True)
    # experiment3()
    experiment4()
