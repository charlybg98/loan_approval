# -*- coding: utf-8 -*-
"""
Module for handling the data loading, preprocessing, and feature engineering tasks.
It prepares the data for both training and inference and handles the preprocessor
save/load functionality.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataLoader:
    """
    The DataLoader class is responsible for loading the dataset, splitting it into
    training and test sets, and applying preprocessing transformations. It also saves
    and loads the preprocessor for future inference.
    """

    def __init__(
        self,
        file_path,
        target_column=None,
        test_size=0.2,
        random_state=42,
        inference_mode=False,
    ):
        """
        Initializes the DataLoader with file path and configuration options.

        Args:
            file_path (str): Path to the CSV file.
            target_column (str): Name of the target column for training.
            test_size (float): Proportion of the dataset to include in the test split (for training mode).
            random_state (int): Seed for reproducibility.
            inference_mode (bool): Flag to indicate whether data is being prepared for inference.
                                   If True, target_column will be ignored.
        """
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.inference_mode = inference_mode
        self.data = None
        self.preprocessor_path = "config/preprocessor.pkl"

    def load_data(self):
        """
        Loads the data from the CSV file.

        Returns:
            pandas.DataFrame: The loaded data.
        """
        self.data = pd.read_csv(self.file_path)
        return self.data

    def _save_preprocessor(self, preprocessor):
        """
        Saves the entire preprocessor (ColumnTransformer) to a file for future inference use.

        Args:
            preprocessor (ColumnTransformer): The fitted preprocessor containing the scaler and encoder.
        """
        os.makedirs(os.path.dirname(self.preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, self.preprocessor_path)

    def _load_preprocessor(self):
        """
        Loads the preprocessor (ColumnTransformer) from a file.

        Returns:
            ColumnTransformer: The preprocessor loaded from the file.
        """
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {self.preprocessor_path}. Run training first."
            )
        return joblib.load(self.preprocessor_path)

    def prepare_data(self):
        """
        Prepares the data for training or inference. In training mode, it fits the preprocessor and saves the
        parameters. In inference mode, it loads the saved preprocessor and applies it to the data.

        Returns:
            numpy.ndarray: Processed feature data (X_train, X_test in training mode; X in inference mode).
            numpy.ndarray or None: Target data (y_train, y_test) in training mode; None in inference mode.
        """
        if self.inference_mode:
            return self._prepare_data_for_inference()
        else:
            return self._prepare_data_for_training()

    def _prepare_data_for_training(self):
        """
        Prepares the data for training by fitting the preprocessor and saving the transformation parameters.

        Returns:
            tuple: Prepared training and test data (X_train, X_test, y_train, y_test).
        """
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(include=["number"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                (
                    "cat",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    categorical_cols,
                ),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        self._save_preprocessor(preprocessor)

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    def _prepare_data_for_inference(self):
        """
        Prepares the data for inference by loading the saved preprocessor
        and applying it to the new data.

        Returns:
            numpy.ndarray: Prepared feature data for inference (X).
        """
        preprocessor = self._load_preprocessor()

        X = self.data

        X_prepared = preprocessor.transform(X)

        return np.array(X_prepared)
