import numpy as np
import pandas as pd

from os import path
from sklearn.datasets import fetch_openml

from src.models.model import Dataset
from src.utilities.utils import get_dataset_dir
from src.utilities.settings import DATA, LABELS, LABELS_SMALL, DATA_SMALL


def download_data() -> Dataset:
    """
    Download MNIST dataset
    :return: downloaded dataset
    """
    print("Downloading data")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = y.astype(int)
    X = X.astype(np.float32)
    X = X/255.

    return Dataset(x=X, y=y)


def store_data(data: Dataset, reduced: bool = False):
    """
    Store given dataset
    :param data: dataset
    :param reduced: specifies the type of dataset:
                    if true, store the original dataset with its specific name
                    if false, store the reduced dataset with its specific name
    """
    print("Storing MNIST. ")
    if not reduced:
        data.store(x_name=DATA, y_name=LABELS)
    else:
        data.store(x_name=DATA_SMALL, y_name=LABELS_SMALL)


def load_data(reduced: bool = False) -> Dataset:
    """
    Load dataset from directory
    :param reduced: specifies the type of dataset:
                    if true, load the original dataset
                    if false, load the reduced dataset
    """
    if not reduced:
        x_name = DATA
        y_name = LABELS
    else:
        x_name = DATA_SMALL
        y_name = LABELS_SMALL

    x_file = path.join(get_dataset_dir(), f"{x_name}.csv")
    y_file = path.join(get_dataset_dir(), f"{y_name}.csv")

    print(f"Loading {x_file} ")
    X = pd.read_csv(x_file)

    print(f"Loading {y_file} ")
    y = pd.read_csv(y_file).values.ravel()

    return Dataset(x=X, y=y)