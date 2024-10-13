# -*- coding: utf-8 -*-
"""
Module for training and evaluating the model, with an option to run inference
on the test dataset and generate a submission file for Kaggle competitions.
"""
import json
import os
from datetime import datetime

import pandas as pd

from core.data_loader import DataLoader
from core.evaluation import Evaluator
from core.model import Model


class Trainer:
    """
    The Trainer class orchestrates the training and evaluation process,
    as well as generating predictions and submission files in inference mode.
    """

    def __init__(
        self,
        model_hps_path="config/model_hps.json",
        train_file="data/raw/train.csv",
        test_file="data/raw/test.csv",
        target_column="loan_status",
        inference_mode=False,
    ):
        """
        Initializes the Trainer with file paths and configuration options.

        Args:
            model_hps_path (str): Path to the JSON file containing the best hyperparameters.
            train_file (str): Path to the training data file (CSV).
            test_file (str): Path to the test data file (CSV).
            target_column (str): Name of the target column for training.
            inference_mode (bool): Flag indicating whether to run inference on the test set after training.
        """
        self.model_hps_path = model_hps_path
        self.train_file = train_file
        self.test_file = test_file
        self.target_column = target_column
        self.inference_mode = inference_mode

        self.model_hps = self._load_hyperparameters()

    def _load_hyperparameters(self):
        """
        Loads the best hyperparameters from the JSON file.

        Returns:
            dict: The best hyperparameters for the model.
        """
        with open(self.model_hps_path, "r") as f:
            return json.load(f)

    def train_and_evaluate(self):
        """
        Trains the model with the best hyperparameters, evaluates it on the test set,
        and optionally generates predictions for submission if in inference mode.
        """
        loader = DataLoader(file_path=self.train_file, target_column=self.target_column)
        loader.load_data()
        X_train, X_test, y_train, y_test = loader.prepare_data()

        model = Model(**self.model_hps)
        model.fit(X_train, y_train)

        evaluator = Evaluator()
        y_pred = model.get_model().predict(X_test)
        y_pred_prob = model.get_model().predict_proba(X_test)[:, 1]
        evaluator.evaluate(y_test, y_pred, y_pred_prob)

        if self.inference_mode:
            self.infer_and_submit(model)

    def infer_and_submit(self, model):
        """
        Makes predictions on the test set and generates a submission file with predicted probabilities.

        Args:
            model (Model): The trained model used for inference.
        """
        loader = DataLoader(file_path=self.test_file, inference_mode=True)
        loader.load_data()
        X_inference = loader.prepare_data()

        test_predictions_prob = model.get_model().predict_proba(X_inference)[:, 1]

        submission_df = pd.DataFrame(
            {"id": loader.data["id"], "loan_status": test_predictions_prob}
        )
        submission_filename = f"data/submissions/submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs(os.path.dirname(submission_filename), exist_ok=True)
        submission_df.to_csv(submission_filename, index=False)

        print(f"Submission file created: {submission_filename}")


if __name__ == "__main__":
    trainer = Trainer(inference_mode=True)
    trainer.train_and_evaluate()
