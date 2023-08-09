from os import path
from src.utilities.settings import DATASETS, IMAGES


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


def get_images_dir() -> str: #todo not needed anyomre
    """
    :return: path to images directory
    """
    return path.join(get_root_dir(), IMAGES)


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
