from __future__ import annotations

import os
import joblib
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Evaluation:
    def __init__(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        self.y_pred = y_pred
        self.y_real = y_true
        self.acc_score = accuracy_score(y_true, y_pred)

    def acc_eval(self):
        """
        return the accuracy score of the predictions
        :return:
        """
        print("-----Model Evaluations:-----")
        print('Accuracy score: {}'.format(self.acc_score))
    
    def conf_mat(self):
        """
        plot the confusion matrix
        :return:
        """
        cmat = confusion_matrix(y_true=self.y_real, y_pred=self.y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cmat,
            display_labels=[str(n) for n in range(10)]
        )
        disp.plot(cmap='Blues_r')


class EvaluatedModel:
    def __init__(self, model, name, test_eval: Evaluation):
        self.model = model
        self.model_name = name
        self.test_eval = test_eval

    def save_evaluation(self):
        """
        save the model and the train set and test set evaluations
        :return:
        """
        if not os.path.exists('../models_evaluation'):
            os.mkdir('../models_evaluation')

        joblib.dump(self, f'../models_evaluation/{self.model_name}.pkl')

    @staticmethod
    def load_evaluation(model_name):
        """
        load previously saved model and its evaluations
        :param model_name:
        :return:
        """
        return joblib.load(f'../models_evaluation/{model_name}.pkl')


class ModelsComparison:
    def __init__(self):
        # load all the final models
        b_nb_model = EvaluatedModel.load_evaluation("beta_naivebayes")
        knn_model = EvaluatedModel.load_evaluation("knearestneighbors")
        rf_model = EvaluatedModel.load_evaluation("randomforest")
        svc_lin_model = EvaluatedModel.load_evaluation("svclinear")
        svc_pol_model = EvaluatedModel.load_evaluation("svcpolynomial")
        svc_rbf_model = EvaluatedModel.load_evaluation("svcrbf")
        self.models = [svc_lin_model, svc_pol_model, svc_rbf_model, rf_model, knn_model, b_nb_model]
        self.all_performances = None

    def rank_model(self):
        """
        ranking based on accuracy score of the model
        :return:
        """
        min_acc_score = None
        acc_score = None

        for i, m in enumerate(self.models):
            if i == 0:
                min_acc_score = m
                acc_score = m.test_eval.acc_score
            if acc_score < m.test_eval.acc_score:
                min_acc_score = m
                acc_score = m.test_eval.acc_score

        return min_acc_score.model, acc_score

    def get_performance_df(self):
        """
        return a pd.DataFrame of the models and their accuracy score
        :return:
        """
        perf_records = []
        for model in self.models:
            record = {
                "model": model.model_name,
                "Accuracy Score": model.test_eval.acc_score}
            perf_records.append(record)

        self.all_performances = pd.DataFrame.from_records(data=perf_records)

        return self.all_performances

    def performance_plot(self):
        """
        plot the score
        :return:
        """
        sns.set(rc={"figure.figsize": (8, 4)})
        plot = sns.lineplot(data=self.all_performances, x="model", y="Accuracy Score", hue="model", style="model", palette="bright", markers=True)
        plot.tick_params(axis="x", rotation=90)

        return plot
