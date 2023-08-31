from typing import List
from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture

from src.models.dataset import Dataset
from src.models.clustering import ClusteringModel
from src.utilities.settings import RANDOM_SEED


N_JOBS = -1


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
        self.model_type = GaussianMixture(max_iter=1000, random_state=RANDOM_SEED)
