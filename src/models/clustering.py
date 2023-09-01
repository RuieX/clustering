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
from sklearn.metrics import rand_score, confusion_matrix
from sklearn.decomposition import PCA
from IPython.display import display

from src.models.dataset import Dataset
from src.utilities.utils import get_results_dir, get_images_dir


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
        # dictionaries keyed by PCA dimensions
        components_results = {}
        components_bestmodels = {}

        # initialize with the lowest rand scores possible
        best_score_glb = -1

        for n in tqdm_notebook(self._n_components, desc=''):
            tqdm.write(f'Processing PCA dimension: {n}')
            data_pca = self.data.reduction_PCA(n_comps=n).normalize()  # applying PCA reduction to dataset
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
            json.dump(self.results, file)

        print(f"Saving {results_bestmodel_name}")
        with open(results_bestmodel_name, 'w') as fl:
            json.dump(self.results_bestmodels, fl)

        print(f"Saving {bestmodel_name}")
        with open(bestmodel_name, 'wb') as f:
            pickle.dump(self.best_model, f)

    @property
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

    @property
    def results_bestmodels(self) -> Dict[int, Dict[str, ModelType | float]]:
        """
        Get the best models for each PCA dimension.
        """
        self._is_evaluated()
        return self._results_bestmodels

    @property
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
            k: {k2: v2[k] for k2, v2 in self.results.items()}
            for k in self.results[list(self.results.keys())[0]]
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
            x2 = [float(n) for n in self.results_bestmodels.keys()]
            y2 = [res2[result] for res2 in self.results_bestmodels.values()]
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

        # Save the image
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


def _get_labels(data: Dataset, model_name: str, best_model_info: dict):
    """
    Get cluster labels for the given dataset and specified clustering model.
    The clustering model and its corresponding parameters are specified based on the given model_name.
    :param data: dataset
    :param model_name: name of the clustering model to use (MeanShift, SpectralClustering, GaussianMixture)
    :param best_model_info: dictionary containing information about the best clustering model
    :return: cluster labels assigned to the data points
    """
    num_components = best_model_info["n_components"]
    data = data.reduction_PCA(n_comps=num_components).normalize()
    best_model: ModelType = best_model_info["model"]

    match model_name:
        case "GaussianMixture":
            labels = best_model.fit_predict(data.x)
        case "MeanShift":
            labels = best_model.labels_
        case "NormalizedCut":
            labels = best_model.labels_
        case _:
            print("The model can only be GaussianMixture, MeanShift, or NormalizedCut")
            return
    return labels


def plot_cluster_frequencies(data: Dataset, model_name: str, best_model_info: dict, save: bool = False):
    """
    Plot the distribution of cluster frequencies in a dataset.
    It creates a histogram to display how data points are distributed across each cluster.
    Display only max_clusters number of clusters.
    :param data: dataset
    :param model_name: name of the clustering model
    :param best_model_info: dictionary containing information about the best clustering model
    :param save: Whether to save the plots as images.
    :return:
    """
    # Get cluster labels
    labels = _get_labels(data=data, model_name=model_name, best_model_info=best_model_info)

    # Determine the number of unique clusters
    n_clusters = len(set(labels))
    max_clusters = 20

    # Sort the labels in descending order
    sorted_labels = sorted(labels, reverse=True)

    cmap = plt.cm.get_cmap('tab20')
    plt.figure(figsize=(16, 6))

    # Handle cases: number of clusters more or less than max_clusters
    if n_clusters <= max_clusters:
        for i in range(n_clusters):
            cls_labels = [label for label in labels if label == i]
            color = i % cmap.N
            plt.hist(cls_labels, bins=[i-0.5 for i in range(n_clusters + 1)], color=cmap(color), label=f'Cluster {i}')
    else:
        # Plot the highest max_clusters clusters
        for i in range(max_clusters - 1):
            cls_labels = [label for label in sorted_labels if label == i]
            color = i % cmap.N
            plt.hist(cls_labels, bins=[i - 0.5 for i in range(max_clusters)], color=cmap(color), label=f'Cluster {i}')

        # Plot the remaining cluster
        other_clusters = [label for label in sorted_labels if label >= max_clusters - 1]
        color = (max_clusters - 1) % cmap.N
        plt.hist(other_clusters, bins=[max_clusters-1-0.5, max_clusters-0.5], color=cmap(color), label='Others')

    plt.xlabel('Cluster')
    plt.ylabel('Frequency')
    plt.title(f'Cluster Frequencies: {n_clusters} clusters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the image
    if save:
        if not os.path.exists(get_images_dir()):
            os.mkdir(get_images_dir())
        file_name = os.path.join(get_images_dir(), f"{model_name}_cluster_frequencies.png")
        plt.savefig(file_name)

    plt.show()


def plot_cluster_composition(data: Dataset, model_name: str, best_model_info: dict, save: bool = False):
    """
    Plot the composition of clusters in terms of actual digit labels.
    Calculates and displays cluster composition analysis, including probabilities of clusters assigned to each digit.
    Note: the percentage for a cluster to be considered correctly able to recognise a digit is set to 50%
    :param data: dataset
    :param model_name: name of the clustering model
    :param best_model_info: dictionary containing information about the best clustering model
    :param save: Whether to save the plots as images.
    :return:
    """
    # Get cluster labels and actual labels
    best_labels = _get_labels(data=data, model_name=model_name, best_model_info=best_model_info)
    actual_labels = data.y

    # Calculate the confusion matrix
    confusion = confusion_matrix(best_labels, actual_labels)
    # Normalize the confusion matrix to get probabilities
    cluster_probs = confusion / confusion.sum(axis=1, keepdims=True)

    # Display cluster-to-digit probabilities in a DataFrame if MeanShift model
    # Otherwise, plot a heatmap
    if model_name == "MeanShift":
        cluster_df = pd.DataFrame(cluster_probs[:, :10])
        cluster_df = cluster_df.applymap(lambda x: f'{x:.3f}')
        cluster_df.columns = [f'Digit {i}' for i in range(0, 10)]
        cluster_df.index.name = "Cluster"
        display(cluster_df)
    else:
        plt.figure(figsize=(8, 10))
        sns.heatmap(cluster_probs[:, :10], annot=True, fmt=".3f", cmap='inferno', square=True)
        plt.xlabel('Actual Digit')
        plt.ylabel('Cluster')
        plt.title('Cluster Composition Analysis (Probabilities)')

        # Save the image
        if save:
            if not os.path.exists(get_images_dir()):
                os.mkdir(get_images_dir())
            file_path = os.path.join(get_images_dir(), f"{model_name}_cluster_composition.png")
            plt.savefig(file_path)

        plt.show()

    # Calculate percentage of clusters focused on each digit (0 to 9)
    digit_focus = (cluster_probs[:, :10] >= 0.5).mean(axis=0) * 100  # set to 50%
    underperforming_pct = 100 - digit_focus.sum()

    print("Percentage of clusters focused on each digit:")
    for digit, pct in enumerate(digit_focus):
        print(f"For digit {digit}: {pct:.3f}%")

    print(f"Underperforming Clusters (distributed across multiple digits): {underperforming_pct:.3f}%")


def plot_reconstructed_images(data: Dataset, model_name: str, best_model_info: dict):
    """
    Plot some reconstructed images for each cluster using the original data points in each cluster.
    :param data: dataset
    :param model_name: name of the clustering model
    :param best_model_info: dictionary containing information about the best clustering model
    :return:
    """
    # Get cluster labels
    best_labels = _get_labels(data=data, model_name=model_name, best_model_info=best_model_info)

    # Fit and reduce dataset dimensionality using PCA with the specified number of components on the original data
    pca = PCA(n_components=best_model_info['n_components'])
    pca.fit(data.x)
    data_pca = pca.transform(data.x)

    unique_clusters = np.unique(best_labels)  # Get unique clusters
    max_clusters = 20  # Set the maximum number of clusters to visualize

    # Loop over clusters and visualize images for each cluster
    for idx, cluster_id in enumerate(unique_clusters):
        if idx >= max_clusters:
            print(f"Cluster visualization limit reached. Remaining clusters won't be displayed.")
            break

        # Find the indices of data points belonging to the current cluster
        cluster_indices = np.where(best_labels == cluster_id)[0]
        cluster_data = data_pca[cluster_indices]  # Extract original data points

        # Select a random subset of data points to display
        n_images_to_display = min(3, len(cluster_data))  # Limit to the number of available images
        random_indexes = random.sample(range(len(cluster_data)), n_images_to_display)
        random_data = cluster_data[random_indexes]

        # Perform PCA inverse transform on the random subset to get original data points
        random_original_data_points = pca.inverse_transform(random_data)

        # Visualize the reconstructed images
        fig, axes = plt.subplots(1, n_images_to_display, figsize=(10, 2))

        for i, ax in enumerate(axes):
            ax.imshow(random_original_data_points[i].reshape(28, 28), cmap='plasma')
            ax.axis('off')

        plt.suptitle(f"Cluster {cluster_id}", fontsize=16)
        plt.show()


def plot_model_means(data: Dataset, model_name: str, best_model_info: dict, save: bool = False):
    """
    Plot the mean of each cluster in the dataset using PCA.
    :param data: dataset
    :param model_name: name of the clustering model
    :param best_model_info: dictionary containing information about the best clustering model
    :param save: Whether to save the plots as images.
    :return:
    """
    # Get cluster labels using the best model
    best_labels = _get_labels(data=data, model_name=model_name, best_model_info=best_model_info)

    # Fit and reduce dataset dimensionality using PCA with the specified number of components on the original data
    pca = PCA(n_components=best_model_info['n_components'])
    pca.fit(data.x)
    data_pca = pca.transform(data.x)

    unique_clusters = np.unique(best_labels)  # Get unique clusters
    num_clusters_to_plot = min(20, len(unique_clusters))  # Limit to 20 or the number of clusters
    num_images_per_row = 3  # Number of cluster means to display in each row of subplots

    # Calculate the number of rows for and create subplot layout
    rows, cols = (num_clusters_to_plot + (num_images_per_row - 1)) // num_images_per_row, num_images_per_row
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))

    for idx, cluster_id in enumerate(unique_clusters[:num_clusters_to_plot]):
        row = idx // num_images_per_row
        col = idx % num_images_per_row

        # Find the indices of data points belonging to the current cluster
        cluster_indices = np.where(best_labels == cluster_id)[0]
        cluster_data_pca = data_pca[cluster_indices]  # Extract transformed data points

        # Calculate the mean of the cluster in the PCA space
        cluster_mean = np.mean(cluster_data_pca, axis=0)

        # Inverse transform the cluster mean to the original data space
        reconstructed_mean = pca.inverse_transform(cluster_mean)

        ax = axes[row, col] if rows > 1 else axes[col]
        ax.imshow(reconstructed_mean.reshape(28, 28), cmap='viridis')
        ax.set_title(f"Cluster {cluster_id} Mean")
        ax.axis('off')

    # Hide any remaining empty subplots
    for idx in range(len(unique_clusters), rows * num_images_per_row):
        row = idx // num_images_per_row
        col = idx % num_images_per_row
        axes[row, col].axis('off')

    if len(unique_clusters) > num_clusters_to_plot:
        print("Note: Only showing the first 20 clusters. Rest are not displayed.")

    plt.tight_layout()

    # Save the image
    if save:
        if not os.path.exists(get_images_dir()):
            os.mkdir(get_images_dir())
        file_path = os.path.join(get_images_dir(), f"{model_name}_cluster_means.png")
        plt.savefig(file_path)

    plt.show()
