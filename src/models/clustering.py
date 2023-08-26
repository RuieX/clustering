import os
import time
import json
import numpy as np
import pandas as pd

from abc import ABC
from typing import Dict, List, TypeVar, Optional
from tqdm import tqdm
from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import rand_score
from matplotlib import pyplot as plt

from src.models.dataset import Dataset
from src.utilities.utils import get_results_dir, get_images_dir
from src.utilities.settings import IMG_EXT, RANDOM_SEED


N_JOBS = -1
ModelType = TypeVar("ModelType", MeanShift, SpectralClustering, GaussianMixture)


class ClusteringModelEvaluation(ABC):
    """
    ClusteringModelEvaluation, an abstract class to implement multiple evaluation
    over multiple hyper-parameters and dimensionality

    This class automatize different clustering models evaluation over a different combination of:
        - hyperparameter (size of kernel, number of clusters)
        - number of components
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """

    def __init__(self, data: Dataset, n_components: List[int], hyperparam_vals: List[int | float]):
        """
        :param data: dataset for evaluation
        :param n_components: list of number of components to evaluate
        :param hyperparam_vals: list of model's hyperparameter value
        """
        self.data: Dataset = data
        self._n_components: List[int] = n_components
        self._hyperparam_values: List[int | float] = hyperparam_vals

        self._evaluated: bool = False
        self._best_model: Optional[ModelType] = None  # instance of ModelType but can also be None
        self._results: Dict[int | float, Dict[int, Dict[str, float]]] = dict()

    def _evaluate(self, model: ModelType, model_name, hyperparam_name: str):
        """
        Evaluate a ClusteringModel over all combination of
            - number of components used
            - hyperparameter
        Results are organized in a dictionary providing:
            - number of clusters found
            - random index score of any model
            - evaluation time
        :param model: implementation of a specific clustering model
        """
        components = {}  # n_components : dictionary keyed by hyperparameter

        for n in tqdm(self._n_components, desc=''):
            tqdm.write(f'Processing number of components: {n}')
            data_d = self.data.make_pca(n_comps=n).rescale()
            best_score = -1  # initialize with the lowest rand scores
            hyperparameters = {}

            for k in tqdm(self._hyperparam_values, desc='', leave=False):
                tqdm.write(f'Processing {hyperparam_name} value: {k}')
                model = model.set_params(**{hyperparam_name: k})
                t1 = time.perf_counter()
                model.fit(data_d)
                t2 = time.perf_counter()
                elapsed = t2 - t1

                score = rand_score(self.data.y, model.labels_)
                results = {
                    'score': score,
                    'n_clusters': model.n_clusters,
                    'time': elapsed
                }

                if score > best_score:
                    best_score = score
                    self._best_model = model

                hyperparameters[k] = results
            components[n] = hyperparameters
        self._results = components
        self._evaluated = True

        # Save the results to a JSON file using the constructed filename
        self._save_result(model_name=model_name, hyperparam_name=hyperparam_name)

    def _is_evaluated(self):
        """
        Check if model was evaluated,
            it raises an exception if it hasn't
        """
        if not self._evaluated:
            raise Exception("Model has not been evaluated yet.")
        return self._evaluated

    def _save_result(self, model_name: str, hyperparam_name: str):
        """
        :return:
        """
        if not os.path.exists(get_results_dir()):
            os.mkdir(get_results_dir())

        filename = os.path.join(get_results_dir(), f"{model_name}_{hyperparam_name}.json")

        print(f"Saving {filename}")
        with open(filename, 'w') as file:
            json.dump(self.results(), file)

    def results(self) -> Dict[float, Dict[int, Dict[str, float]]]:
        """
        Provides results of evaluation in a dictionary format ( kernel size : number of components : clusters, score )
        """
        self._is_evaluated()
        return self._results

    def best_model(self) -> ModelType:
        """
        Returns best model in the evaluation
        """
        self._is_evaluated()
        return self._best_model

# todo will come to this part later about PLOT
#     def _plot(self, title: str, res: str, y_label: str,
#               save: bool = False, file_name: str = 'graph'):
#         """
#         Plot a graph foreach different kernel used:
#             - x axes: number of component
#             - y axes: stats (number of clusters / score / time)
#         :param title: graph title
#         :param res: weather score or number of cluster or time
#         :param y_label: name for ordinates axes
#         :param save: if to save the graph to images directory
#         :param file_name: name of stored file
#         """
#
#         # transform
#         #   components     : hyperparameter : results
#         #   hyperparameter : components     : results
#         inverted_dictionary = {
#             k: {k2: v2[k] for k2, v2 in self.results.items()}
#             for k in self.results[list(self.results.keys())[0]]
#         }
#
#         for kernel, dims in inverted_dictionary.items():
#
#             x = []  # number of components
#             y = []  # result
#
#             for nc, out in dims.items():
#                 x.append(nc)
#                 y.append(out[res])
#
#             # Plot the points connected by a line
#             plt.plot(x, y, '-o', label=f'{kernel}  ')
#
#         # Add a legend
#         plt.legend(bbox_to_anchor=(1, 1), title=self.HYPERPARAMETER, loc='upper left', borderaxespad=0.)
#
#         # Set the x and y axis labels
#         plt.title(title)
#         plt.xlabel('Number of components')
#         plt.ylabel(y_label)
#
#         # Show the plot
#         if save:
#             makedir(get_images_dir())
#             file_name = path.join(get_images_dir(), f"{file_name}.{IMG_EXT}")
#             plt.savefig(file_name)
#
#         # SAve the plot
#         plt.show()
#
#     def plot_score(self, save=False, file_name='accuracy'):
#         """
#         Plot score graph
#         :save: if to save the graph to images directory
#         :file_name: name of stored file
#         """
#         self._plot(title="Random Index Score", res=self.SCORE,
#                    y_label='Score', save=save, file_name=file_name)
#
#     def plot_n_clusters(self, save=False, file_name='n_clusters'):
#         """
#         Plot n_cluster graph
#         :save: if to save the graph to images directory
#         :file_name: name of stored file
#         """
#         self._plot(title="Varying Cluster Number", res=self.N_CLUSTERS,
#                    y_label='NClusters', save=save, file_name=file_name)
#
#     def plot_time(self, save=False, file_name='time'):
#         """
#         Plot time execution graph
#         :save: if to save the graph to images directory
#         :file_name: name of stored file
#         """
#         self._plot(title="Elapsed Execution Time", res=self.TIME,
#                    y_label='Time', save=save, file_name=file_name)


def load_result(model_name: str, hyperparam_name: str) -> Dict[float, Dict[int, Dict[str, float]]]:
    """

    :param model_name:
    :param hyperparam_name:
    :return:
    """
    # Specify the path to your JSON file
    filename = os.path.join(get_results_dir(), f"{model_name}_{hyperparam_name}.json")

    # Open the JSON file in read mode
    print(f"Loading {filename}")
    with open(filename, 'r') as json_file:
        result = json.load(json_file)

    return result


# MEAN SHIFT

class MeanShiftEvaluation(ClusteringModelEvaluation):
    """
    This class automatize different MeanShiftCluster models evaluation over a different combination of:
        - kernel size
        - number of components
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """
    model_name = "MeanShift"
    hyperparameter_name = "bandwidth"

    def evaluate(self):
        """
        Evaluate MeanShift Clustering over all combination of
            - number of components used
            - kernel dimension (bandwidth)
        """
        self._evaluate(MeanShift(n_jobs=N_JOBS),
                       model_name=self.model_name,
                       hyperparam_name=self.hyperparameter_name)


# NORMALIZED CUT

class NormalizedCutEvaluation(ClusteringModelEvaluation):
    """
    This class automatize different NormalizedCutClustering models evaluation over a different combination of:
        - k (number of clusters)
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """
    model_name = "NormalizedCut"
    hyperparameter_name = "n_clusters"

    def evaluate(self):
        """
        Evaluate NormalizedCut Clustering over all combination of
            - number of components used
            - k (number of clusters)
        """
        self._evaluate(SpectralClustering(n_jobs=N_JOBS, random_state=RANDOM_SEED),
                       model_name=self.model_name,
                       hyperparam_name=self.hyperparameter_name)


# MIXTURE GAUSSIAN

class MixtureGaussianEvaluation(ClusteringModelEvaluation):
    """
    This class automatize different NormalizedCutClustering models evaluation over a different combination of:
        - k (number of clusters)
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """
    model_name = "GaussianMixture"
    hyperparameter_name = "n_components"

    def evaluate(self):
        """
        Evaluate MixtureGaussian Clustering over all combination of
            - number of components used
            - k (number of clusters)
        """
        self._evaluate(GaussianMixture(max_iter=200, random_state=RANDOM_SEED),
                       model_name=self.model_name,
                       hyperparam_name=self.hyperparameter_name)
