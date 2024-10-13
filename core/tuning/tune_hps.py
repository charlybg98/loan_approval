# -*- coding: utf-8 -*-
"""
Module for hyperparameter tuning using Optuna. It loads the training dataset,
runs the hyperparameter optimization, and saves the best hyperparameters for later use.
"""
import json
import os

import optuna
from sklearn.metrics import roc_auc_score

from core.data_loader import DataLoader
from core.model import Model


class HyperparameterTuner:
    """
    The HyperparameterTuner class handles the process of tuning hyperparameters using
    the Optuna framework. It loads the dataset, defines the objective function for
    optimization, and saves the best hyperparameters.
    """

    def __init__(self, data_file, target_column, test_size=0.2, random_state=42):
        """
        Initializes the HyperparameterTuner with data and configuration options.

        Args:
            data_file (str): Path to the training data file (CSV).
            target_column (str): Name of the target column for training.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed for reproducibility.
        """
        self.data_file = data_file
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.best_params_path = "config/model_hps.json"

    def load_data(self):
        """
        Loads and prepares the data using the DataLoader class.

        Returns:
            tuple: A tuple containing training and test data (X_train, X_test, y_train,
            y_test) as numpy arrays.
        """
        loader = DataLoader(
            file_path=self.data_file,
            target_column=self.target_column,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        loader.load_data()
        return loader.prepare_data()

    def objective(self, trial):
        """
        Defines the objective function for Optuna hyperparameter tuning.

        Args:
            trial (optuna.Trial): A trial object from Optuna, which suggests hyperparameter
            values.

        Returns:
            float: The AUC score of the model on the test set, to be maximized by Optuna.
        """
        X_train, X_test, y_train, y_test = self.load_data()

        n_estimators = trial.suggest_int("n_estimators", 300, 1000)
        max_depth = trial.suggest_int("max_depth", 5, 10)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        gamma = trial.suggest_float("gamma", 0, 1)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 1e-1, log=True)

        model = Model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

        model.fit(X_train, y_train)

        y_pred_prob = model.get_model().predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)

        return auc

    def tune_hyperparameters(self, n_trials=50):
        """
        Runs the hyperparameter optimization using Optuna and saves the best hyperparameters.

        Args:
            n_trials (int): The number of trials to run in the optimization.

        Returns:
            dict: The best hyperparameters found during the tuning process.
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)

        best_params = study.best_params
        self._save_best_hyperparameters(best_params)

        print(f"Best hyperparameters: {best_params}")
        return best_params

    def _save_best_hyperparameters(self, best_params):
        """
        Saves the best hyperparameters found by Optuna to a JSON file.

        Args:
            best_params (dict): The best hyperparameters found by Optuna.
        """
        os.makedirs(os.path.dirname(self.best_params_path), exist_ok=True)
        with open(self.best_params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4)


if __name__ == "__main__":
    tuner = HyperparameterTuner(
        data_file="data/raw/train.csv",
        target_column="loan_status",
        test_size=0.2,
        random_state=42,
    )

    tuner.tune_hyperparameters(n_trials=1000)
