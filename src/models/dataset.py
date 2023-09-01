import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.utilities.utils import get_dataset_dir


class Dataset:
    """
    Represents a dataset with features and corresponding labels as a tuple of:
     - feature data as a Pandas DataFrame
     - labels as a NumPy array
    The class allows you to create, manipulate, and save datasets.
    """
    def __init__(self, x: pd.DataFrame, y: np.ndarray):
        """
        :param x: feature data
        :param y: labels
        """
        # data and labels must have the same length
        if len(x) != len(y):
            raise Exception(f"X has length {len(x)}, while y has {len(y)}")

        self._x: pd.DataFrame = x
        self._y: np.ndarray = np.array(y)

    @property
    def x(self) -> pd.DataFrame:
        """
        :return: feature data
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

    def normalize(self) -> 'Dataset':
        """
        Normalizes the features of a dataset using Min-Max scaling.
        Scales the features to the range [0, 1] to ensure that all features have similar scales.
        :return: a new Dataset object with normalized features using Min-Max scaling.
        """
        new_x = pd.DataFrame(MinMaxScaler().fit_transform(self.x), columns=self.features)
        return Dataset(
            x=new_x,
            y=self.y
        )

    def reduction_PCA(self, n_comps: int) -> 'Dataset':
        """
        Reduces the dimensionality of a dataset to the given number of principal components
        using Principal Component Analysis (PCA).
        :param n_comps: number of principal components to retain.
        it must be a positive integer less than the number of features in the original dataset.
        :return: a new dataset with reduced dimensionality.
        """
        if n_comps < 0:
            raise Exception(
                "Number of components must a be positive integer")

        n_features = len(self.x.columns)
        if n_comps >= n_features:
            raise Exception(
                f"Number of components must be less than the number of features in the original dataset {n_features}")

        # return new dataset with reduced dimensionality
        return Dataset(
            x=pd.DataFrame(PCA(n_components=n_comps).fit_transform(self.x)),
            y=self.y
        )

    # SAVE
    def save(self, x_name: str = 'data_X', y_name: str = 'data_y'):
        """
        Stores the dataset in dataset directory
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
