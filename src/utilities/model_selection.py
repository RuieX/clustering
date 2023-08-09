import time
import os
import joblib

from loguru import logger
from sklearn.model_selection import GridSearchCV

# todo idk if needed


def model_selector(estimator, properties, scoring, cv, verbose, jobs, x_train, y_train):
    """
    model selection using GridSearchCV, and print the execution time
    :param estimator:
    :param properties:
    :param scoring:
    :param cv:
    :param verbose:
    :param jobs:
    :param x_train:
    :param y_train:
    :return:
    """
    start_time = time.time()
    tuned_model = GridSearchCV(estimator, properties, scoring=scoring, cv=cv,
                               return_train_score=True, verbose=verbose, n_jobs=jobs)
    tuned_model.fit(x_train, y_train)
    logger.info("--- %s seconds ---" % (time.time() - start_time))

    return tuned_model


def save_model(model, model_name):
    """
    save tuned model
    :param model:
    :param model_name:
    :return:
    """
    if not os.path.exists('../models'):
        os.mkdir('../models')
    joblib.dump(model, f'../models/{model_name}.pkl')


def load_model(model_name):
    """
    load previously saved tuned model
    :param model_name:
    :return:
    """
    return joblib.load(f"../models/{model_name}.pkl")
