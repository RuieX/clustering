import os
import time
import copy
import json
import pickle
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
    over multiple hyperparameters and dimensionality

    This class automatize different clustering models evaluation over a different combination of:
        - hyperparameter (size of kernel, number of clusters)
        - number of components
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """
    model_type: Optional[ModelType] = None
    model_name: str = 'model'
    hyperparameter_name: str = 'hyperparameter'

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
        self._results: Dict[int, Dict[float, Dict[str, float]]] = dict()
        self._results_bestmodels: Dict[int, Dict[str, ModelType | float]] = dict()
        self._best_model: Dict[str, ModelType | float] = dict()

    def evaluate(self):
        """
        Evaluate a ClusteringModel over all combination of
            - number of components used
            - hyperparameter
        Results are organized in a dictionary providing:
            - number of clusters found
            - random index score of any model
            - evaluation time
        :param model: implementation of a specific clustering model TODO delete this line
        """
        components_results = {}  # n_components : dictionary keyed by hyperparameter
        components_bestmodels = {}

        best_score_glb = -1  # initialize with the lowest rand scores
        for n in tqdm(self._n_components, desc=''):
            tqdm.write(f'Processing number of components: {n}')
            data_pca = self.data.make_pca(n_comps=n).rescale()
            best_score_lcl = -1
            hyperparameters = {}

            for k in tqdm(self._hyperparam_values, desc='', leave=False):
                tqdm.write(f'Processing {self.hyperparameter_name} value: {k}')
                model = self.model_type.set_params(**{self.hyperparameter_name: k})
                t1 = time.perf_counter()
                labels = model.fit_predict(data_pca.x)
                t2 = time.perf_counter()
                elapsed = t2 - t1

                score = rand_score(data_pca.y, labels)
                results = {
                    'score': score,
                    'n_clusters': len(set(labels)),
                    'time': elapsed
                }

                # locally to value n
                if score > best_score_lcl:
                    best_score_lcl = score
                    components_bestmodels[n] = {
                        'model': copy.deepcopy(model),
                        f'{self.hyperparameter_name}': k,
                        'score': score,
                        'n_clusters': len(set(labels)),
                        'time': elapsed
                    }

                # overall
                if score > best_score_glb:
                    best_score_glb = score
                    self._best_model = {
                        'model': copy.deepcopy(model),
                        'n_components': n,
                        f'{self.hyperparameter_name}': k,
                        'score': score,
                        'n_clusters': len(set(labels)),
                        'time': elapsed
                    }

                hyperparameters[k] = results
            components_results[n] = hyperparameters
        self._results = components_results
        self._results_bestmodels = components_bestmodels
        self._evaluated = True

        # Save the results to a JSON file using the constructed filename
        self._save_result()

    def _is_evaluated(self):
        """
        Check if model was evaluated,
            it raises an exception if it hasn't
        """
        if not self._evaluated:
            raise Exception("Model has not been evaluated yet.")
        return self._evaluated

    def _save_result(self):
        """
        :return:
        """
        if not os.path.exists(get_results_dir()):
            os.mkdir(get_results_dir())

        results_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_result.json")
        results_bestmodel_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_result_bestmodels.pkl")
        bestmodel_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_bestmodel.pkl")

        print(f"Saving {results_name}")
        with open(results_name, 'w') as file:
            json.dump(self.results(), file)

        print(f"Saving {results_bestmodel_name}")
        with open(results_bestmodel_name, 'wb') as fl:
            pickle.dump(self.results_bestmodels(), fl)

        print(f"Saving {bestmodel_name}")
        with open(bestmodel_name, 'wb') as f:
            pickle.dump(self.best_model(), f)

    def results(self) -> Dict[int, Dict[float, Dict[str, float]]]:
        """
        Provides results of evaluation in a dictionary format ( kernel size : number of components : clusters, score )
        """
        self._is_evaluated()
        return self._results

    def results_bestmodels(self) -> Dict[int, Dict[str, ModelType | float]]:
        """

        """
        self._is_evaluated()
        return self._results_bestmodels

    def best_model(self) -> Dict[str, ModelType | float]:
        """
        Returns best model in the evaluation
        """
        self._is_evaluated()
        return self._best_model


# todo fixed but not tried

    
    def load_results(self):
        """
        """
        # Specify the path to your JSON file
        results_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_result.json")
        results_bestmodel_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_result_bestmodels.pkl")
        bestmodel_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_bestmodel.pkl")

        # Open the JSON file in read mode
        print(f"Loading {results_name}")
        with open(results_name, 'r') as file:
            results = json.load(file)

        print(f"Loading {results_bestmodel_name}")
        with open(results_bestmodel_name, 'rb') as fl:
            result_bestmodels = pickle.load(fl)

        print(f"Loading {bestmodel_name}")
        with open(bestmodel_name, 'rb') as f:
            best_model = pickle.load(f)

        self._results = results
        self._results_bestmodels = result_bestmodels
        self._best_model = best_model
        self._evaluated = True





# todo plot haven't started seeing this



    def _plot(self, title: str, res: str, y_label: str,
              save: bool = False, file_name: str = 'graph'):
        """
        Plot a graph foreach different kernel used:
            - x axes: number of component
            - y axes: stats (number of clusters / score / time)
        :param title: graph title
        :param res: weather score or number of cluster or time
        :param y_label: name for ordinates axes
        :param save: if to save the graph to images directory
        :param file_name: name of stored file
        """

        # transform
        #   components     : hyperparameter : results
        #   hyperparameter : components     : results
        inverted_dictionary = {
            k: {k2: v2[k] for k2, v2 in self.results().items()}
            for k in self.results()[list(self.results().keys())[0]]
        }

        for kernel, dims in inverted_dictionary.items():

            x = []  # number of components
            y = []  # result

            for nc, out in dims.items():
                x.append(nc)
                y.append(out[res])

            # Plot the points connected by a line
            plt.plot(x, y, '-o', label=f'{kernel}  ')

        # Add a legend
        plt.legend(bbox_to_anchor=(1, 1), title=self.hyperparameter_name, loc='upper left', borderaxespad=0.)

        # Set the x and y-axis labels
        plt.title(title)
        plt.xlabel('Number of components')
        plt.ylabel(y_label)

        # Show the plot
        if save:
            if not os.path.exists(get_images_dir()):
                os.mkdir(get_images_dir())
            file_name = os.path.join(get_images_dir(), f"{file_name}.{IMG_EXT}")
            plt.savefig(file_name)

        # SAve the plot
        plt.show()

    def plot_score(self, save=False, file_name='accuracy'):
        """
        Plot score graph
        :save: if to save the graph to images directory
        :file_name: name of stored file
        """
        self._plot(title="Random Index Score", res='score',
                   y_label='Score', save=save, file_name=file_name)

    def plot_n_clusters(self, save=False, file_name='n_clusters'):
        """
        Plot n_cluster graph
        :save: if to save the graph to images directory
        :file_name: name of stored file
        """
        self._plot(title="Varying Cluster Number", res='n_clusters',
                   y_label='NClusters', save=save, file_name=file_name)

    def plot_time(self, save=False, file_name='time'):
        """
        Plot time execution graph
        :save: if to save the graph to images directory
        :file_name: name of stored file
        """
        self._plot(title="Elapsed Execution Time", res='time',
                   y_label='Time', save=save, file_name=file_name)


# MEAN SHIFT

class MeanShiftEvaluation(ClusteringModelEvaluation):
    """
    This class automatize different MeanShiftCluster models evaluation over a different combination of:
        - kernel size
        - number of components
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """
    model = "MeanShift"
    hyperparameter_name = "bandwidth"

    def __init__(self, data: Dataset, n_components: List[int], hyperparam_vals: List[int | float]):
        super().__init__(data, n_components, hyperparam_vals)
        self.hyperparameter = self.hyperparameter_name
        self.model_name = self.model
        self.model_type = MeanShift(n_jobs=N_JOBS)


# NORMALIZED CUT

class NormalizedCutEvaluation(ClusteringModelEvaluation):
    """
    This class automatize different NormalizedCutClustering models evaluation over a different combination of:
        - k (number of clusters)
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """
    model = "NormalizedCut"
    hyperparameter_name = "n_clusters"

    def __init__(self, data: Dataset, n_components: List[int], hyperparam_vals: List[int | float]):
        super().__init__(data, n_components, hyperparam_vals)
        self.hyperparameter = self.hyperparameter_name
        self.model_name = self.model
        self.model_type = SpectralClustering(n_jobs=N_JOBS, random_state=RANDOM_SEED)


# MIXTURE GAUSSIAN

class MixtureGaussianEvaluation(ClusteringModelEvaluation):
    """
    This class automatize different NormalizedCutClustering models evaluation over a different combination of:
        - k (number of clusters)
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """
    model = "GaussianMixture"
    hyperparameter_name = "n_components"

    def __init__(self, data: Dataset, n_components: List[int], hyperparam_vals: List[int | float]):
        super().__init__(data, n_components, hyperparam_vals)
        self.hyperparameter = self.hyperparameter_name
        self.model_name = self.model
        self.model_type = GaussianMixture(max_iter=200, random_state=RANDOM_SEED)
