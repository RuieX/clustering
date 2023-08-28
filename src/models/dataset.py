import time
from abc import abstractmethod, ABC
import os
from random import sample
from typing import Iterator, Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from statistics import mean, mode
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.utilities.utils import get_dataset_dir, get_images_dir
# from assignment_3.clustering.utils import digits_histogram, plot_digit, plot_mean_digit, plot_cluster_frequencies_histo

"""

This module provides basic classes to implement clustering logic
 - Dataset, a class to provide a unique view for data and labels
 - DataClusterSplit, a class to split data into clusters

"""


""" DATASET """


class Dataset:
    """
    This class represent a Dataset as a tuple:
     - feature metrix as a pandas dataframe
     - label vector as an array
    """

    def __init__(self, x: pd.DataFrame, y: np.ndarray):
        """
        :param x: feature matrix
        :param y: label vector
        """

        # data and labels must have the same length
        if len(x) != len(y):
            raise Exception(f"X has length {len(x)}, while y has {len(y)}")

        self._x: pd.DataFrame = x
        self._y: np.ndarray = np.array(y)

    @property
    def x(self) -> pd.DataFrame:
        """
        :return: feature matrix
        """
        return self._x

    @property
    def y(self) -> np.ndarray:
        """
        :return: labels
        """
        return self._y

    @property
    def features(self) -> List[str]:
        """
        :return: features name
        """
        return list(self.x.columns)

    def __len__(self) -> int:
        """
        :return: rows in the feature matrix
        """
        return len(self.x)

    def __iter__(self) -> Iterator[DataFrame | ndarray]:
        """
        :return: unpacked fields
        """
        return iter([self.x, self.y])

    def __str__(self) -> str:
        """
        :return: class stringify
        """
        return f"[Length: {len(self)}; Features: {len(self.x.columns)}]"

    def __repr__(self) -> str:
        """
        :return: class representation
        """
        return str(self)

    def rescale(self) -> 'Dataset':
        """
        Rescales rows and columns in interval [0, 1]
        """
        new_x = pd.DataFrame(MinMaxScaler().fit_transform(self.x), columns=self.features)
        return Dataset(
            x=new_x,
            y=self.y
        )

    def make_pca(self, n_comps: int) -> 'Dataset':
        """
         reduces the dataset's features to a specified number of components
        Applies principal component analysis to the feature space
        :param n_comps: number of components for the reduced output dataset
            an integrity check is made to check if the required number of components is feasible
        :return: dataset with reduced number of components
        """

        # integrity checks
        if n_comps < 0:
            raise Exception("Number of components must be positive")

        actual_comps = len(self.x.columns)
        if n_comps >= actual_comps:
            raise Exception(f"Number of components must be less than the actual number of components{actual_comps}")

        # return new object
        return Dataset(
            x=pd.DataFrame(PCA(n_components=n_comps).fit_transform(self.x)),
            y=self.y
        )

    # SAVE
    # todo do i need default str?
    def save(self, x_name: str = 'dataX', y_name: str = 'datay'):
        """
        Stores the dataset in datasets directory
        :param x_name: name of feature file
        :param y_name: name of labels file
        """

        if not os.path.exists(get_dataset_dir()):
            os.mkdir(get_dataset_dir())

        x_out = os.path.join(get_dataset_dir(), f"{x_name}.csv")
        y_out = os.path.join(get_dataset_dir(), f"{y_name}.csv")

        print(f"Saving {x_out}")
        self.x.to_csv(x_out, index=False)

        print(f"Saving {y_out}")
        pd.DataFrame(self.y).to_csv(y_out, index=False)

# TODO prolly useless since i do it outside this class
#
#     def plot_digits(self):
#         """
#         Plots all digits in the dataset
#         """
#         for i in range(len(self)):
#             pixels = np.array(self.X.iloc[i])
#             plot_digit(pixels=pixels)
#
#     def plot_mean_digits(self):
#         """
#         Plots mean of all digits in the dataset
#         """
#         plot_mean_digit(X=self.X)


# CLUSTER DATA SPLIT

# class DataClusterSplit:
#     """
#     Provide an interface to split a dataset given clustering index
#     """
#
#     def __init__(self, data: Dataset, index: np.ndarray):
#         """
#
#         :param data: dataset to be split
#         :param index: clustering index
#         """
#         self._clusters: Dict[int, Dataset] = self._split_dataset(data=data, index=index)
#
#         # materialized view of cluster actual_label - true_label for score evaluation
#         self._cluster_idx, self._true_label = self._materialize_indexes()
#
#     # DUNDER
#
#     def __str__(self) -> str:
#         """
#         Return string representation for the object:
#         :return: stringify Data Clustering Split
#         """
#         return f"ClusterDataSplit [Data: {self.total_instances}, Clusters: {self.n_cluster}, " \
#                f"Mean-per-Cluster: {self.mean_cardinality:.3f}, Score: {self.rand_index_score:.3f}] "
#
#     def __repr__(self) -> str:
#         """
#         Return string representation for the object:
#         :return: stringify Data Clustering Split
#         """
#         return str(self)
#
#     # STATS
#
#     @property
#     def clusters(self) -> Dict[int, Dataset]:
#         """
#         Return dataset split by clusters in format cluster_id : cluster data
#         :return: data split in cluster
#         """
#         return self._clusters
#
#     @property
#     def n_cluster(self) -> int:
#         """
#         Returns the number of clusters found
#         :return: number of clusters
#         """
#         return len(self.clusters)
#
#     @property
#     def clusters_cardinality(self) -> Dict[int, int]:
#         """
#         Return the number of objects for each cluster
#         :return: mapping cluster_id : number of elements
#         """
#         return {k: len(v) for k, v in self.clusters.items()}
#
#     @property
#     def total_instances(self) -> int:
#         """
#         Returns the total number of points among all clusters
#         :return: total number of instances among all clusters
#         """
#         return sum(self.clusters_cardinality.values())
#
#     @property
#     def clusters_frequencies(self) -> Dict[int, int]:
#         """
#         Return the frequencies of cluster cardinality
#         :return: cluster cardinality frequencies
#         """
#         lengths = list(self.clusters_cardinality.values())
#         return {x: lengths.count(x) for x in lengths}
#
#     @property
#     def mean_cardinality(self) -> float:
#         """
#         Return average cluster cardinality
#         :return: average cluster cardinality
#         """
#         return mean(self.clusters_cardinality.values())
#
#     # ALTER THE CLUSTER
#
#     def get_sub_clusters(self, a: int | None = None, b: int | None = None) -> DataClusterSplit:
#         """
#         Returns a new DataClusterSplit with cluster cardinalities in given range [a, b]
#         :param a: cardinality lower bound, zero if not given
#         :param b: cardinality upper bound, maximum cardinality if not given
#         """
#         if a is None:  # lower-bound to zero
#             a = 0
#         if b is None:  # upper-bound to maximum cardinality
#             b = max(self.clusters_cardinality.values())
#
#         dcs = DataClusterSplit(  # generating new fake DataClusterSplit
#             data=Dataset(x=pd.DataFrame([0]), y=np.array([0])),
#             index=np.array([0])
#         )
#         dcs._clusters = {  # setting new datas satisfying length bounds
#             k: v for k, v in self.clusters.items()
#             if a <= len(v) <= b
#         }
#         dcs._cluster_idx, dcs._true_label = dcs._materialize_indexes()
#         return dcs
#
#     @staticmethod
#     def _split_dataset(data: Dataset, index: np.ndarray) -> Dict[int, Dataset]:
#         """
#         Split the Dataset in multiple given a certain index
#         :param data: dataset to split
#         :param index: indexes for split
#         :return: dataset split according to index
#         """
#         values = list(set(index))  # get unique values
#         return {
#             v: Dataset(
#                 x=data.X[index == v].reset_index(drop=True),
#                 y=data.y[index == v]
#             )
#             for v in values
#         }
#
#     # PLOTS
#
#     def plot_frequencies_histo(self, save: bool = False, file_name: str = 'frequencies'):
#         """
#         Plot frequencies in a histogram
#         :save: if to save the graph to images directory
#         :file_name: name of stored file
#         """
#
#         plot_cluster_frequencies_histo(frequencies=self.clusters_frequencies, save=save, file_name=file_name)
#
#     def plot_mean_digit(self, sample_out: int | None = None):
#         """
#         Plots mean digit foreach cluster
#         :param sample_out: number of elements to print uniformly sampled
#         """
#
#         vals = list(self.clusters.values())
#
#         if sample_out is not None:
#             vals = sample(vals, sample_out)
#
#         for c in vals:
#             freq = {x: list(c.y).count(x) for x in c.y}
#             freq = dict(sorted(freq.items(), key=lambda x: -x[1]))  # sort by values
#             print(f"[Mode {mode(c.y)}: {freq}] ")
#             plot_mean_digit(X=c.X)
#
#     # SCORE
#
#     def _materialize_indexes(self) -> Tuple[List[int], List[int]]:
#         """
#         Provides list of clusters and corresponding labels to evaluate scores
#         """
#
#         cluster_idx = [item for sublist in [
#             [idx] * len(data) for idx, data in self.clusters.items()
#         ] for item in sublist]
#
#         true_labels = np.concatenate([
#             data.y for _, data in self.clusters.items()
#         ]).ravel().tolist()
#
#         return cluster_idx, true_labels
#
#     @property
#     def rand_index_score(self) -> float:
#         """
#         :return: clustering rand index score
#         """
#         return rand_score(labels_true=self._true_label, labels_pred=self._cluster_idx)
