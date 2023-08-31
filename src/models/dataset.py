import os
from typing import Iterator, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.utilities.utils import get_dataset_dir


# DATASET
# todo documentation, remove useless functions

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
