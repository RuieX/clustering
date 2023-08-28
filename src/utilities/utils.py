import os
from os import path
from src.utilities.settings import DATASETS, IMAGES, RESULTS


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


def get_results_dir() -> str:
    """
    :return: path to dataset directory
    """
    return path.join(get_root_dir(), RESULTS)


def get_images_dir() -> str:
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
