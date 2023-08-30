"""
This module implements general purpose functions
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from typing import List, Dict
from matplotlib import pyplot as plt

from src.utilities.utils import get_images_dir


def chunks(lst: List, n: int) -> np.array:
    """
    Split given list in a matrix with fixed-length rows length
    :param lst: list to split
    :param n: length of sublist
    :return: matrix with n rows
    """

    def chunks_():
        """
        Auxiliary function to exploit yield operator property
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    list_len = len(lst)

    # list length must me a multiple of the required length for sublist
    if list_len % n != 0:
        raise Exception(f"Cannot split list of {list_len} in {n} rows")

    sub_lists = list(chunks_())

    return np.array(
        [np.array(sl) for sl in sub_lists]
    )


""" PLOTS """


def plot_digit(pixels: np.array, save: bool = False,
               file_name: str = "digit"):
    """
    Plot a figure given a square matrix array,
        each cell represent a grey-scale pixel with intensity 0-1
    :param pixels: intensity of pixels
    :param save: true for storing the image
    :param file_name: name file if stored
    """
    SIZE = 28
    fig, ax = plt.subplots(1)
    pixels = chunks(lst=pixels, n=SIZE)
    ax.imshow(pixels, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if save:
        if not os.path.exists(get_images_dir()):
            os.mkdir(get_images_dir())
        file = os.path.join(get_images_dir(), f"{file_name}.png")
        print(f"Saving {file} ")
        plt.savefig(file)  # return allows inline-plot in notebooks

    plt.show()


def plot_mean_digit(X: pd.DataFrame, save: bool = False,
                    file_name: str = "mean_digit"):
    """
    Plots the average figure of a certain number of images
    :param X: set of images
    :param save: if true, image is stored
    :param file_name: name of file if stored
    """

    pixels = np.mean(X, axis=0)
    plot_digit(pixels=pixels, save=save, file_name=file_name)


def digits_histogram(labels: pd.DataFrame | np.ndarray,
                     save: bool = False, file_name: str = "plot"):
    """
    Plot distribution of labels in a dataset given its labels

    :param labels: collection with labels
    :param save: if true, the image is stored in the directory
    :param file_name: name of file if stored (including extension)
    """

    # type-check and casting
    if type(labels) == np.ndarray:
        labels = pd.DataFrame(labels)

    # digits count
    digits: Dict[str, int] = {
        k[0]: v for k, v in labels.value_counts().to_dict().items()
    }

    # plot
    fig, ax = plt.subplots(1)
    ax.bar(list(digits.keys()), digits.values(), edgecolor='black')
    ax.set_xticks(range(10))
    ax.set_title('Digits distribution')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Counts')

    if save:
        if not os.path.exists(get_images_dir()):
            os.mkdir(get_images_dir())
        file = os.path.join(get_images_dir(), f"{file_name}.png")
        print(f"Saving {file} ")
        plt.savefig(file)  # return allows inline-plot in notebooks

    plt.show()


def plot_cluster_frequencies_histo(frequencies: Dict[int, int],
                                   save: bool = False, file_name: str = 'frequencies'):
    """
    Plot clusters frequencies in a histogram
    :save: if to save the graph to images directory
    :file_name: name of stored file
    """

    fig, ax = plt.subplots(1)

    ax.bar(list(frequencies.keys()), frequencies.values(), edgecolor='black')

    # Title and axes
    ax.set_title('Clusters cardinality')
    ax.set_xlabel('Cluster dimension')
    ax.set_ylabel('Occurrences')

    if save:
        if not os.path.exists(get_images_dir()):
            os.mkdir(get_images_dir())
        out_file = os.path.join(get_images_dir(), f"{file_name}.png")
        print(f"Saving {out_file} ")
        plt.savefig(out_file)
    plt.show()
