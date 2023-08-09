import os
import numpy as np
import pandas as pd

# todo not needed in this assignment


class TrainTestSplit:
    """
    class used to represent the train and test split of the original dataset
    """
    def __init__(self, x_train: pd.DataFrame, x_test: pd.DataFrame,
                 y_train: np.ndarray, y_test: np.ndarray):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def to_csv(self, dir_path: str):
        """
        save the split data
        :param dir_path:
        :return:
        """
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        self.x_train.to_csv(f'{dir_path}/x_train.csv', index=False)
        self.x_test.to_csv(f'{dir_path}/x_test.csv', index=False)
        pd.DataFrame(self.y_train).to_csv(f'{dir_path}/y_train.csv', index=False)
        pd.DataFrame(self.y_test).to_csv(f'{dir_path}/y_test.csv', index=False)

    @staticmethod
    def from_csv_directory(dir_path: str) -> "TrainTestSplit":
        """
        load previously saved split data
        :param dir_path:
        :return:
        """
        x_train = pd.read_csv(f'{dir_path}/x_train.csv')
        x_test = pd.read_csv(f'{dir_path}/x_test.csv')

        # The y datasets are only one column
        y_train = pd.read_csv(f'{dir_path}/y_train.csv',).iloc[:, 0].values
        y_test = pd.read_csv(f'{dir_path}/y_test.csv').iloc[:, 0].values

        return TrainTestSplit(x_train, x_test, y_train, y_test)


def get_train_subset(x_train, y_train, size):
    """
    get a subset using stratified sampling
    :param x_train:
    :param y_train:
    :param size:
    :return:
    """
    whole_data = x_train.copy()
    whole_data["Number"] = y_train.copy()

    # to get balanced classes
    data_small = whole_data.groupby('Number', group_keys=False).apply(lambda x: x.sample(int(size/10)))

    y_data_small = data_small["Number"].copy()
    x_data_small = data_small.drop(columns=['Number'])

    return x_data_small, y_data_small
