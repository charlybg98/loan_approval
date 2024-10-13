# -*- coding: utf-8 -*-
"""
Module for defining the model architecture and managing the model's training
and prediction processes.
"""
from xgboost import XGBClassifier


class Model:
    """
    The Model class encapsulates the machine learning model, handles training,
    and provides functionality for making predictions and returning model metrics.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        colsample_bynode=1,
        reg_alpha=0,
        reg_lambda=1,
        min_child_weight=1,
        gamma=0,
        scale_pos_weight=1,
        verbosity=1,
        objective="binary:logistic",
        booster="gbtree",
        random_state=42,
    ):
        """
        Initializes the Model with hyperparameters.

        Args:
            n_estimators (int): Number of boosting rounds. Default is 100.
            max_depth (int): Maximum depth of a tree. Default is 3.
            learning_rate (float): Boosting learning rate (step size shrinkage). Default is 0.1.
            subsample (float): Subsample ratio of the training instances. Default is 1.
            colsample_bytree (float): Subsample ratio of columns when constructing each tree. Default is 1.
            colsample_bylevel (float): Subsample ratio of columns for each level. Default is 1.
            colsample_bynode (float): Subsample ratio of columns for each node split. Default is 1.
            reg_alpha (float): L1 regularization term on weights. Default is 0.
            reg_lambda (float): L2 regularization term on weights. Default is 1.
            min_child_weight (float): Minimum sum of instance weight (hessian) needed in a child. Default is 1.
            gamma (float): Minimum loss reduction required to make a split. Default is 0.
            scale_pos_weight (float): Balancing of positive and negative weights. Default is 1.
            verbosity (int): Verbosity of the output. Default is 1.
            objective (str): Learning task and objective (binary classification). Default is 'binary:logistic'.
            booster (str): Booster type to use ('gbtree', 'gblinear', or 'dart'). Default is 'gbtree'.
            random_state (int): Seed used to generate the random number. Default is 42.
            use_label_encoder (bool): Avoids label encoding warning. Default is False.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.scale_pos_weight = scale_pos_weight
        self.verbosity = verbosity
        self.objective = objective
        self.booster = booster
        self.random_state = random_state
        self.model = None

    def build_model(self):
        """
        Builds the XGBoost model with the defined hyperparameters.

        Returns:
            None
        """
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            scale_pos_weight=self.scale_pos_weight,
            verbosity=self.verbosity,
            objective=self.objective,
            booster=self.booster,
            random_state=self.random_state,
            eval_metric="logloss",
        )

    def fit(self, X_train, y_train):
        """
        Fits the model to the provided training data.

        Args:
            X_train (numpy.ndarray): Training data features.
            y_train (numpy.ndarray): Training data labels.

        Returns:
            None
        """
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts target values for the test dataset.

        Args:
            X_test (numpy.ndarray): Test data features.

        Returns:
            numpy.ndarray: Predicted labels for the test data.

        Raises:
            ValueError: If the model has not been trained before making predictions.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call `fit` first.")
        return self.model.predict(X_test)

    def get_model(self):
        """
        Returns the internal XGBoost model.

        Returns:
            XGBClassifier: The trained XGBoost model.
        """
        return self.model
