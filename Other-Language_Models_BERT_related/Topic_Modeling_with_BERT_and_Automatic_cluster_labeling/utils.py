import numpy as np
import pandas as pd
import random as rn
import re
import nltk
import os

import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

from nltk.corpus import stopwords
from wordcloud import WordCloud

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

pd.set_option("display.max_rows", 600)
pd.set_option("display.max_columns", 500)
pd.set_option("max_colwidth", 400)

import umap  # dimensionality reduction
import hdbscan  # clustering
from functools import partial

# To perform the Bayesian Optimization for searching the optimum hyperparameters,
# we use hyperopt package:
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials
import random


def generate_clusters(
    message_embeddings,
    n_neighbors,
    n_components,
    min_cluster_size,
    min_samples=None,
    random_state=None,
):
    """
    Generate clusters using UMAP and HDBSCAN algorithms.

    Args:
        message_embeddings (np.ndarray): Array of message embeddings.
        n_neighbors (int): Number of neighbors for UMAP algorithm.
        n_components (int): Number of dimensions for UMAP algorithm.
        min_cluster_size (int): Minimum number of samples in a cluster for HDBSCAN algorithm.
        min_samples (int, optional): Minimum number of samples for HDBSCAN algorithm.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        hdbscan.HDBSCAN: Clustering model.

    """
    umap_embeddings = (
        umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric="cosine",
            random_state=random_state,
        )
    ).fit_transform(message_embeddings)

    clusters = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        gen_min_span_tree=True,
        cluster_selection_method="eom",
    ).fit(umap_embeddings)

    return clusters


## Hyperopt


def score_clusters(clusters, prob_thresh0ld=0.05):
    """
    Score the clusters based on their labels and probabilities.

    Args:
        clusters (hdbscan.HDBSCAN): Clustering model.
        prob_threshold (float, optional): Probability threshold for considering a sample as an outlier.
            Default is 0.05.

    Returns:
        int: Number of unique cluster labels.
        float: Cost score representing the fraction of samples with probabilities below the threshold.

    """
    cluster_labels = clusters.labels_

    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)

    cost = np.count_nonzero(clusters.probabilities_ < prob_thresh0ld) / total_num

    return label_count, cost


def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperparameter optimization.

    Args:
        params (dict): Dictionary of hyperparameters.
        embeddings (np.ndarray): Array of message embeddings.
        label_lower (int): Lower bound for the number of unique cluster labels.
        label_upper (int): Upper bound for the number of unique cluster labels.

    Returns:
        dict: Dictionary containing the loss value, number of unique cluster labels, and status.

    """
    clusters = generate_clusters(
        embeddings,
        n_neighbors=params["n_neighbors"],
        n_components=params["n_components"],
        min_cluster_size=params["min_cluster_size"],
        random_state=params["random_state"],
    )

    label_count, cost = score_clusters(clusters, prob_thresh0ld=0.05)

    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 1.0
    else:
        penalty = 0

    loss = cost + penalty

    return {"loss": loss, "label_count": label_count, "status": STATUS_OK}


# Then minimize the objective function over the hyperparameter search space using the
# Tree-structured Parzen Estimator (TPE) algorithm:
def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayesian search on hyperparameter space using hyperopt

    Arguments:
        embeddings: embeddings to use
        space: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', and 'random_state' and
               values that use built-in hyperopt functions to define
               search spaces for each
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters
        max_evals: int, maximum number of parameter combinations to try

    Saves the following to instance variables:
        best_params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'min_samples', and 'random_state' and
               values associated with lowest cost scenario tested
        best_clusters: HDBSCAN object associated with lowest cost scenario
                       tested
        trials: hyperopt trials object for search

    """

    trials = Trials()
    fmin_objective = partial(
        objective,
        embeddings=embeddings,
        label_lower=label_lower,
        label_upper=label_upper,
    )

    best = fmin(
        fmin_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )

    best_params = space_eval(space, best)
    print("best:")
    print(best_params)
    print(f"label count: {trials.best_trial['result']['label_count']}")

    best_clusters = generate_clusters(
        embeddings,
        n_neighbors=best_params["n_neighbors"],
        n_components=best_params["n_components"],
        min_cluster_size=best_params["min_cluster_size"],
        random_state=best_params["random_state"],
    )

    return best_params, best_clusters, trials
