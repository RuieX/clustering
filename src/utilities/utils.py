import os
import numpy as np
import pandas as pd

from os import path
from sklearn.datasets import fetch_openml

from src.models.model import Dataset
from src.utilities.settings import DATASETS, IMAGES, DATA, LABELS,LABELS_SMALL, DATA_SMALL


def get_root_dir() -> str:
    """
    :return: path to root directory
    """
    return str(path.dirname(path.abspath(path.join(__file__, "../"))))


def get_dataset_dir() -> str:
    """
    :return: path to dataset directory
    """
    return path.join(get_root_dir(), DATASETS)


def get_images_dir() -> str:  #todo not needed anyomre?
    """
    :return: path to images directory
    """
    return path.join(get_root_dir(), IMAGES)


def has_files_in_dir(dir_path):
    """
    Check if there are any files in given directory
    :param dir_path:
    :return: true if there are files, false otherwise
    """
    items_in_dir = os.listdir(dir_path)
    files_in_dir = [item for item in items_in_dir if os.path.isfile(os.path.join(dir_path, item))]

    return len(files_in_dir) > 0


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

    return Dataset(x=X,y=y)


# def nomakedirifnotexist(path_: str): #todo delete once done
#     tfidf_dir = os.path.join(samples_dir, "tfidf")
#     if not os.path.exists(tfidf_dir):
#         os.mkdir(tfidf_dir)
#
#     tfidf_results_path = os.path.join(tfidf_dir, "tfidf_docs.pkl")
#
#     try:
#         os.makedirs(path_)
#         print(f"Created directory {path_} ")
#     except OSError:
#         pass
