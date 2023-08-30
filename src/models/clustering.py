import os
import time
import copy
import json
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC
from typing import Dict, List, TypeVar, Optional
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import rand_score

from src.models.dataset import Dataset
from src.utilities.utils import get_results_dir, get_images_dir
from src.utilities.settings import RANDOM_SEED


N_JOBS = -1
ModelType = TypeVar("ModelType", MeanShift, SpectralClustering, GaussianMixture)


class ClusteringModel(ABC):
    """
    Abstract class for evaluating various clustering models (MeanShift, SpectralClustering, GaussianMixture).
    This class generalizes the evaluation process across different models by considering combinations of:
        - PCA dimensions
        - model specific hyperparameters
        (MeanShift: bandwidth, SpectralClustering: n_clusters, GaussianMixture: n_components)
    The class also provides methods for retrieving the best models and plotting evaluation results.
    """
    model_type: Optional[ModelType] = None
    model_name: str = 'model'
    hyperparameter_name: str = 'hyperparameter'

    def __init__(self, data: Dataset, n_components: List[int], hyperparam_vals: List[int | float]):
        """
        Initialize the ClusteringModel instance.
        :param data: dataset for evaluation
        :param n_components: list of PCA dimensions to evaluate
        :param hyperparam_vals: list of model's hyperparameter values
        """
        self.data: Dataset = data
        self._n_components: List[int] = n_components
        self._hyperparam_values: List[int | float] = hyperparam_vals

        self._results: Dict[int, Dict[float, Dict[str, float]]] = dict()
        self._results_bestmodels: Dict[int, Dict[str, ModelType | float]] = dict()
        self._best_model: Dict[str, ModelType | float] = dict()
        self._evaluated: bool = False

    def evaluate(self):
        """
        Evaluate the given clustering model over all combination of PCA dimensions and hyperparameters.
        All results will be stored in self._results providing:
            - number of clusters
            - random index score
            - evaluation time
        The models with the best score for each PCA dimensions will be stored in self._results_bestmodels.
        The best model overall will be stored in self._best_model.
        """
        components_results = {}  # dictionaries keyed by PCA dimensions
        components_bestmodels = {}

        best_score_glb = -1  # initialize with the lowest rand scores possible

        for n in tqdm_notebook(self._n_components, desc=''):
            tqdm.write(f'Processing PCA dimension: {n}')
            data_pca = self.data.make_pca(n_comps=n).rescale()  # applying PCA to dataset
            best_score_lcl = -1
            hyperparameters = {}  # dictionary keyed by hyperparameter value

            for k in tqdm_notebook(self._hyperparam_values, desc='', leave=False):
                tqdm.write(f'PCA dimension: {n} - {self.hyperparameter_name} value: {k}')
                model = self.model_type.set_params(**{self.hyperparameter_name: k})  # changing model's parameters
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

                # for local best model (over hyperparameter values)
                if score > best_score_lcl:
                    best_score_lcl = score
                    components_bestmodels[n] = {
                        f'{self.hyperparameter_name}': k,
                        'score': score,
                        'n_clusters': len(set(labels)),
                        'time': elapsed
                    }

                # for overall best model
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
            tqdm.write("")
            components_results[n] = hyperparameters
        self._results = components_results
        self._results_bestmodels = components_bestmodels
        self._evaluated = True

        # Save the results to files
        self._save_result()

    def _is_evaluated(self):
        """
        Check if the model has been evaluated.
        Raises an exception otherwise
        """
        if not self._evaluated:
            raise Exception("Model has not been evaluated yet.")
        return self._evaluated

    def _save_result(self):
        """
        Save the evaluation results and best models to JSON and pickle files.
        """
        if not os.path.exists(get_results_dir()):
            os.mkdir(get_results_dir())

        results_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_result.json")
        results_bestmodel_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_result_bestmodels.json")
        bestmodel_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_bestmodel.pkl")

        print(f"Saving {results_name}")
        with open(results_name, 'w') as file:
            json.dump(self.results(), file)

        print(f"Saving {results_bestmodel_name}")
        with open(results_bestmodel_name, 'w') as fl:
            json.dump(self.results_bestmodels(), fl)

        print(f"Saving {bestmodel_name}")
        with open(bestmodel_name, 'wb') as f:
            pickle.dump(self.best_model(), f)

    def results(self) -> Dict[int, Dict[float, Dict[str, float]]]:
        """
        Get the evaluation results as a dictionary, where the keys are the PCA dimensions.
        Each key corresponds to a nested dictionary with hyperparameter values as keys containing:
            - number of clusters
            - random index score
            - evaluation time.
        """
        self._is_evaluated()
        return self._results

    def results_bestmodels(self) -> Dict[int, Dict[str, ModelType | float]]:
        """
        Get the best models for each PCA dimension.
        """
        self._is_evaluated()
        return self._results_bestmodels

    def best_model(self) -> Dict[str, ModelType | float]:
        """
        Get the best model in the evaluation.
        """
        self._is_evaluated()
        return self._best_model

    def load_results(self):
        """
        Load the evaluation results and best models from files.
        """
        results_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_result.json")
        results_bestmodel_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_result_bestmodels.json")
        bestmodel_name = os.path.join(
            get_results_dir(), f"{self.model_name}_{self.hyperparameter_name}_bestmodel.pkl")

        print(f"Loading {results_name}")
        with open(results_name, 'r') as file:
            results = json.load(file)

        print(f"Loading {results_bestmodel_name}")
        with open(results_bestmodel_name, 'r') as fl:
            result_bestmodels = json.load(fl)

        print(f"Loading {bestmodel_name}")
        with open(bestmodel_name, 'rb') as f:
            best_model = pickle.load(f)

        self._results = results
        self._results_bestmodels = result_bestmodels
        self._best_model = best_model
        self._evaluated = True

    # PLOT

    def _plot(self, title: str, result: str, y_label: str, ax=None, highlight_best=False):
        """
        Generate a line plot PCA dimension vs a result metric (score/number of clusters/time)
        :param title: Title of the plot.
        :param result: The specific result metric to plot (e.g., score, n_clusters, time).
        :param y_label: Label for the y-axis.
        :param ax: Axes object for plotting (optional).
        :param highlight_best: Whether to highlight the best model results.
        """
        # Switching keys PCA dimensions and hyperparameter for easier access to result of each hyperparameter
        inverted_dict = {
            k: {k2: v2[k] for k2, v2 in self.results().items()}
            for k in self.results()[list(self.results().keys())[0]]
        }

        for param, dim in inverted_dict.items():
            x = [float(n) for n in dim.keys()]  # PCA dimensions
            y = [res[result] for res in dim.values()]  # result

            # Create a new Axes instance if ax is not provided
            if ax is None:
                fig, ax = plt.subplots()

            # Plot the points on the specified Axes
            if highlight_best:
                ax.plot(x, y, '-o', label=f'{param}', alpha=0.5)
            else:
                ax.plot(x, y, '-o', label=f'{param}')

        # Highlight best model for each PCA dimension
        if highlight_best:
            x2 = [float(n) for n in self.results_bestmodels().keys()]
            y2 = [res2[result] for res2 in self.results_bestmodels().values()]
            ax.plot(x2, y2, 'o', markersize=12, color='gold', label='Best Model')

        # Set the x and y-axis labels
        sns.set_style("whitegrid")
        plt.title(title)
        plt.xlabel('PCA dimension')
        plt.ylabel(y_label)

        # Add legend
        ax.legend(bbox_to_anchor=(1, 1), title=self.hyperparameter_name, loc='upper left', borderaxespad=0.)

    def _plot_with_highlight(self, title: str, result: str, y_label: str, highlight_best=False,
                             save: bool = False, file_name: str = 'plot'):
        """
        Generate two side-by-side line plots: one with original results and another with best model results highlighted.
        :param title: Title of the plots.
        :param result: The specific result metric to plot (e.g., score, n_clusters, time).
        :param y_label: Label for the y-axis.
        :param highlight_best: Whether to highlight the best model results.
        :param save: Whether to save the plots as images.
        :type file_name: File name to save as.
        """
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Plot with original results
        self._plot(title=title, result=result, y_label=y_label, ax=axs[0], highlight_best=False)

        # Plot with best model results highlighted
        self._plot(title="Best Model "+title, result=result, y_label=y_label, ax=axs[1], highlight_best=highlight_best)

        plt.tight_layout()

        if save:
            if not os.path.exists(get_images_dir()):
                os.mkdir(get_images_dir())
            file_name = os.path.join(get_images_dir(), f"{file_name}.png")
            plt.savefig(file_name)

        # Show the plot
        plt.show()

    def plot_score_with_highlight(self, save=False):
        """
        Plot score vs PCA dimension with highlighted best model results.
        :param save: Whether to save the plots as images.
        """
        self._plot_with_highlight(title="Random Index Score",
                                  result='score',
                                  y_label='Score',
                                  highlight_best=True,
                                  save=save,
                                  file_name=f'{self.model_name}_score')

    def plot_n_clusters_with_highlight(self, save=False):
        """
        Plot n_cluster vs PCA dimension with highlighted best model results.
        :param save: Whether to save the plots as images.
        """
        self._plot_with_highlight(title="Cluster Number",
                                  result='n_clusters',
                                  y_label='No. Clusters',
                                  highlight_best=True,
                                  save=save,
                                  file_name=f'{self.model_name}_n_clusters')

    def plot_time_with_highlight(self, save=False):
        """
        Plot execution time vs PCA dimension with highlighted best model results.
        :param save: Whether to save the plots as images.
        """
        self._plot_with_highlight(title="Execution Time",
                                  result='time',
                                  y_label='Time',
                                  highlight_best=True,
                                  save=save,
                                  file_name=f'{self.model_name}_time')


# MEAN SHIFT

class MeanShiftEvaluation(ClusteringModel):
    """
    Class for evaluating MeanShift clustering models using combination of
    PCA dimensions and hyperparameter bandwidth values.
    """
    model = "MeanShift"
    hyperparameter_name = "bandwidth"

    def __init__(self, data: Dataset, n_components: List[int], hyperparam_vals: List[int | float]):
        """
        Initialize a MeanShift evaluation instance using SpectralClustering.
        :param data: The dataset for evaluation.
        :param n_components: List of PCA dimensions to evaluate.
        :param hyperparam_vals: List of hyperparameter values to evaluate.
        """
        super().__init__(data, n_components, hyperparam_vals)
        self.hyperparameter = self.hyperparameter_name
        self.model_name = self.model
        self.model_type = MeanShift(n_jobs=N_JOBS)


# NORMALIZED CUT

class NormalizedCutEvaluation(ClusteringModel):
    """
    Class for evaluating NormalizedCut clustering models using combination of
    PCA dimensions and hyperparameter n_clusters values.
    """
    model = "NormalizedCut"
    hyperparameter_name = "n_clusters"

    def __init__(self, data: Dataset, n_components: List[int], hyperparam_vals: List[int | float]):
        """
        Initialize a NormalizedCut evaluation instance using SpectralClustering.
        :param data: The dataset for evaluation.
        :param n_components: List of PCA dimensions to evaluate.
        :param hyperparam_vals: List of hyperparameter values to evaluate.
        """
        super().__init__(data, n_components, hyperparam_vals)
        self.hyperparameter = self.hyperparameter_name
        self.model_name = self.model
        self.model_type = SpectralClustering(n_jobs=N_JOBS, random_state=RANDOM_SEED)


# MIXTURE GAUSSIAN

class MixtureGaussianEvaluation(ClusteringModel):
    """
    Class for evaluating GaussianMixture clustering models using combination of
    PCA dimensions and hyperparameter n_components values.
    """
    model = "GaussianMixture"
    hyperparameter_name = "n_components"  # refers to number of clusters, not PCA dimension

    def __init__(self, data: Dataset, n_components: List[int], hyperparam_vals: List[int | float]):
        """
        Initialize a GaussianMixture evaluation instance.
        :param data: The dataset for evaluation.
        :param n_components: List of PCA dimensions to evaluate.
        :param hyperparam_vals: List of hyperparameter values to evaluate.
        """
        super().__init__(data, n_components, hyperparam_vals)
        self.hyperparameter = self.hyperparameter_name
        self.model_name = self.model
        self.model_type = GaussianMixture(max_iter=200, random_state=RANDOM_SEED)
