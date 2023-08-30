"""
This module implements general purpose functions
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from random import sample
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from statistics import mean, mode
from sklearn.metrics import rand_score

from src.models.dataset import Dataset
from src.models.clustering import ModelType
from src.utilities.utils import get_images_dir


def chunks(lst: List, n: int) -> np.array:
    """
    Split given list in a matrix with fixed-length rows length
    :param lst: list to split
    :param n: length of sublist
    :return: matrix with n rows
    """

    def chunks_():
        """
        Auxiliary function to exploit yield operator property
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    list_len = len(lst)

    # list length must be a multiple of the required length for sublist
    if list_len % n != 0:
        raise Exception(f"Cannot split list of {list_len} in {n} rows")

    sub_lists = list(chunks_())

    return np.array(
        [np.array(sl) for sl in sub_lists]
    )


""" PLOTS """


def plot_digit(pixels: np.array, save: bool = False,
               file_name: str = "digit"):
    """
    Plot a figure given a square matrix array,
        each cell represent a grey-scale pixel with intensity 0-1
    :param pixels: intensity of pixels
    :param save: true for storing the image
    :param file_name: name file if stored
    """
    SIZE = 28
    fig, ax = plt.subplots(1)
    pixels = chunks(lst=pixels, n=SIZE)
    ax.imshow(pixels, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if save:
        if not os.path.exists(get_images_dir()):
            os.mkdir(get_images_dir())
        file = os.path.join(get_images_dir(), f"{file_name}.png")
        print(f"Saving {file} ")
        plt.savefig(file)  # return allows inline-plot in notebooks

    plt.show()


def plot_mean_digit(X: pd.DataFrame, save: bool = False,
                    file_name: str = "mean_digit"):
    """
    Plots the average figure of a certain number of images
    :param X: set of images
    :param save: if true, image is stored
    :param file_name: name of file if stored
    """

    pixels = np.mean(X, axis=0)
    plot_digit(pixels=pixels, save=save, file_name=file_name)


def digits_histogram(labels: pd.DataFrame | np.ndarray,
                     save: bool = False, file_name: str = "plot"):
    """
    Plot distribution of labels in a dataset given its labels

    :param labels: collection with labels
    :param save: if true, the image is stored in the directory
    :param file_name: name of file if stored (including extension)
    """

    # type-check and casting
    if type(labels) == np.ndarray:
        labels = pd.DataFrame(labels)

    # digits count
    digits: Dict[str, int] = {
        k[0]: v for k, v in labels.value_counts().to_dict().items()
    }

    # plot
    fig, ax = plt.subplots(1)
    ax.bar(list(digits.keys()), digits.values(), edgecolor='black')
    ax.set_xticks(range(10))
    ax.set_title('Digits distribution')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Counts')

    if save:
        if not os.path.exists(get_images_dir()):
            os.mkdir(get_images_dir())
        file = os.path.join(get_images_dir(), f"{file_name}.png")
        print(f"Saving {file} ")
        plt.savefig(file)  # return allows inline-plot in notebooks

    plt.show()


def plot_cluster_frequencies_histo(frequencies: Dict[int, int],
                                   save: bool = False, file_name: str = 'frequencies'):
    """
    Plot clusters frequencies in a histogram
    :save: if to save the graph to images directory
    :file_name: name of stored file
    """

    fig, ax = plt.subplots(1)

    ax.bar(list(frequencies.keys()), frequencies.values(), edgecolor='black')

    # Title and axes
    ax.set_title('Clusters cardinality')
    ax.set_xlabel('Cluster dimension')
    ax.set_ylabel('Occurrences')

    if save:
        if not os.path.exists(get_images_dir()):
            os.mkdir(get_images_dir())
        out_file = os.path.join(get_images_dir(), f"{file_name}.png")
        print(f"Saving {out_file} ")
        plt.savefig(out_file)
    plt.show()


# CLUSTER DATA SPLIT

class DataClusterSplit:
    """
    Provide an interface to split a dataset given clustering index
    """

    def __init__(self, data: Dataset, best_model: ModelType, model_name: str):
        """
        get clustering index
        :param data: dataset to be split
        :param best_model:
        :param model_name:
        """
        self.data: Dataset = data
        self._best_model: ModelType = best_model
        self._model_name: str = model_name
        labels = self._get_labels()
        self._clusters: Dict[int, Dataset] = self._split_dataset(index=labels)

        # materialized view of cluster actual_label - true_label for score evaluation
        self._cluster_idx, self._true_label = self._materialize_indexes()

    # DUNDER todo

    def __str__(self) -> str:
        """
        Return string representation for the object:
        :return: stringify Data Clustering Split
        """
        return f"ClusterDataSplit [Data: {self.total_instances}, Clusters: {self.n_cluster}, " \
               f"Mean-per-Cluster: {self.mean_cardinality:.3f}, Score: {self.rand_index_score:.3f}] "

    # STATS

    def clusters(self) -> Dict[int, Dataset]:
        """
        Return dataset split by clusters in format cluster_id : cluster data
        :return: data split in cluster
        """
        return self._clusters

    def n_cluster(self) -> int:
        """
        Returns the number of clusters found
        :return: number of clusters
        """
        return len(self.clusters())

    def clusters_cardinality(self) -> Dict[int, int]:
        """
        Return the number of objects for each cluster
        :return: mapping cluster_id : number of elements
        """
        return {k: len(v) for k, v in self.clusters().items()}

    def total_instances(self) -> int:
        """
        Returns the total number of points among all clusters
        :return: total number of instances among all clusters
        """
        return sum(self.clusters_cardinality().values())

    def clusters_frequencies(self) -> Dict[int, int]:
        """
        Return the frequencies of cluster cardinality
        :return: cluster cardinality frequencies
        """
        lengths = list(self.clusters_cardinality().values())
        return {x: lengths.count(x) for x in lengths}

    def mean_cardinality(self) -> float:
        """
        Return average cluster cardinality
        :return: average cluster cardinality
        """
        return mean(self.clusters_cardinality().values())

    def _split_dataset(self, index: np.ndarray) -> Dict[int, Dataset]:
        """
        Split the Dataset in multiple given a certain index
        :param index: indexes for split
        :return: dataset split according to index
        """
        values = list(set(index))  # get unique values
        return {
            v: Dataset(
                x=self.data.x[index == v].reset_index(drop=True),
                y=self.data.y[index == v]
            )
            for v in values
        }

    def _get_labels(self):
        """

        :return:
        """
        match self._model_name:
            case "GaussianMixture":
                labels = self._best_model.predict(self.data.x)
                # n_clusters_ = best_model.get_params()["n_components"]
                # cluster_centers = best_model.means_
            case "MeanShift":
                labels = self._best_model.labels_
                # n_clusters_ = len(np.unique(labels))
                # cluster_centers = best_model.cluster_centers_
            case "NormalizedCut":
                labels = self._best_model.labels_
                # n_clusters_ = best_model.get_params()["n_clusters"]
                # cluster_centers = None
            case _:
                print("The model can only be GaussianMixture, MeanShift, or NormalizedCut")
                return
        return labels

    # ALTER THE CLUSTER
    def get_sub_clusters(self, a: int | None = None, b: int | None = None) -> "DataClusterSplit":
        """
        Returns a new DataClusterSplit with cluster cardinalities in the given range [a, b].

        Args:
            a (int | None): Cardinality lower bound. Default is 0 if not given.
            b (int | None): Cardinality upper bound. Default is maximum cardinality if not given.

        Returns:
            DataClusterSplit: New instance with filtered clusters.
        """
        if a is None:
            a = 0
        if b is None:
            b = max(self.clusters_cardinality().values())

        filtered_clusters = {
            k: v for k, v in self.clusters().items()
            if a <= len(v) <= b
        }

        # Create a new DataClusterSplit instance with filtered clusters
        filtered_dcs = DataClusterSplit(
            data=Dataset(x=pd.DataFrame([0]), y=np.array([0])),
            best_model=self._best_model,
            model_name=self._model_name
        )
        # dcs = DataClusterSplit(  # generating new fake DataClusterSplit
        #     data=Dataset(x=pd.DataFrame([0]), y=np.array([0])),
        #     index=np.array([0])
        # )
        filtered_dcs._clusters = filtered_clusters
        filtered_dcs._cluster_idx, filtered_dcs._true_label = filtered_dcs._materialize_indexes()

        return filtered_dcs

    # SCORE

    def _materialize_indexes(self) -> Tuple[List[int], List[int]]:
        """
        Provides list of clusters and corresponding labels to evaluate scores
        """
        cluster_idx = [item for sublist in [
            [idx] * len(data) for idx, data in self.clusters().items()
        ] for item in sublist]

        true_labels = np.concatenate([
            data.y for _, data in self.clusters().items()
        ]).ravel().tolist()

        return cluster_idx, true_labels

    def rand_index_score(self) -> float:
        """
        :return: clustering rand index score
        """
        return rand_score(labels_true=self._true_label, labels_pred=self._cluster_idx)

    # PLOTS

    def plot_frequencies_histo(self, save: bool = False, file_name: str = 'frequencies'):
        """
        Plot frequencies in a histogram
        :save: if to save the graph to images directory
        :file_name: name of stored file
        """

        plot_cluster_frequencies_histo(frequencies=self.clusters_frequencies(), save=save, file_name=file_name)

    def plot_mean_digit(self, sample_out: int | None = None):
        """
        Plots mean digit foreach cluster
        :param sample_out: number of elements to print uniformly sampled
        """

        vals = list(self.clusters().values())

        if sample_out is not None:
            vals = sample(vals, sample_out)

        for c in vals:
            freq = {x: list(c.y).count(x) for x in c.y}
            freq = dict(sorted(freq.items(), key=lambda x: -x[1]))  # sort by values
            print(f"[Mode {mode(c.y)}: {freq}] ")
            plot_mean_digit(X=c.x)
