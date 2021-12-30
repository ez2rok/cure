# experiment1
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

# experiment2
from icecream import ic
from sklearn.model_selection import train_test_split as tts
import pandas as pd


# import local files
from make_dataset import iris_data, fashion_mnist_data, add_intercept, synthetic_elliptical_data
from cure import CURE
import evaluate as eval


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
    Compare the performance of CURE and many other clustering algorithms on various datasets.
    Much of this code is taken form the sklearn tutorial here.
    https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
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

    datasets = [
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

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
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
                algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(X)

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
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

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plt.text(
                0.99,
                0.01,
                ("%.2fs" % (t1 - t0)).lstrip("0"),
                transform=plt.gca().transAxes,
                size=15,
                horizontalalignment="right",
            )
            plot_num += 1

        if save:
            plt.savefig(outdir + 'compare_clustering.png')
    fig = plt.gcf()
    return fig


def experiment2():

    def get_results(seed, n, d):
        X, y = synthetic_elliptical_data(
            n, d, seed, mu_val=[0, 4], sigma_val=5)[:2]

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
                misclf = eval.misclassification_rate(y_test, y_pred)
                result.append(misclf)
            results.append(result)

        return results

     # initial values
    n = 1300
    d = 2
    n_trials = 5
    seeds = [i for i in range(n_trials)]

    results = np.array([get_results(seed, n, d) for seed in seeds])
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)

    text = [['{:.1f} ± {:.1f}%'.format(m * 100, s * 100) for m, s in zip(m_s, s_s)]
            for m_s, s_s in zip(means, stds)]
    
    ratios = ['{} : 1'.format(i) for i in range(1, 5)]
    clfs = ['KMeans', 'CURE']
    df = pd.DataFrame(text, index=clfs, columns=ratios)
    
    return df

# initial values
seed = 420

#get data
classes = [2, 1]
X, y = iris_data(classes)
X_train, X_test, y_train, y_test = tts(X, y, random_state=seed)




if __name__ == '__main__':
    # experiment1()
    experiment2()
