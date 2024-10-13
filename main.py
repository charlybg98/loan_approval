# -*- coding: utf-8 -*-
import argparse

from core.model import Trainer
from core.tuning import HyperparameterTuner


class LoanApproval:
    """
    The LoanApproval class is the entry point for the Loan Approval Prediction project.
    It handles training, hyperparameter tuning, and inference based on user input.

    Methods:
        parse_arguments(): Parses the command line arguments.
        run(): Executes the workflow based on the parsed arguments.
    """

    def __init__(self):
        """
        Initializes the LoanApprovalApp and sets up argument parsing.
        """
        self.args = self.parse_arguments()

    def parse_arguments(self):
        """
        Parses command line arguments to determine the mode of operation: train, tune, or inference.
        Also supports n_trials for hyperparameter tuning.

        Returns:
            argparse.Namespace: Parsed command line arguments.
        """
        parser = argparse.ArgumentParser(description="Loan Approval Prediction Project")
        parser.add_argument(
            "--mode",
            type=str,
            required=True,
            choices=["train", "tune", "inference"],
            help="Specify the mode to run: train, tune, or inference",
        )
        parser.add_argument(
            "--n_trials",
            type=int,
            default=50,
            help="Number of trials for hyperparameter tuning (used in tune mode). Default is 50.",
        )
        return parser.parse_args()

    def run(self):
        """
        Executes the appropriate workflow based on the command line arguments.

        Raises:
            ValueError: If the specified mode is not recognized.
        """
        if self.args.mode == "train":
            trainer = Trainer(inference_mode=False)
            trainer.train_and_evaluate()

        elif self.args.mode == "tune":
            tuner = HyperparameterTuner(
                data_file="data/raw/train.csv", target_column="loan_status"
            )
            tuner.tune_hyperparameters(n_trials=self.args.n_trials)

        elif self.args.mode == "inference":
            trainer = Trainer(inference_mode=True)
            trainer.train_and_evaluate()

        else:
            raise ValueError(
                "Invalid mode selected. Choose between 'train', 'tune', or 'inference'."
            )


if __name__ == "__main__":
    main = LoanApproval()
    main.run()
