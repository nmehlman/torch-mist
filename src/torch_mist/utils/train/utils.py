import os
import tempfile
from typing import Optional

from torch import nn
import torch


class RunTerminationManager:
    def __init__(
        self,
        early_stopping: bool,
        tolerance: float,
        patience: int,
        warmup_iterations: Optional[int],
        max_iterations: Optional[int],
        verbose: bool = False,
        maximize: bool = False,
        minimize: bool = False,
    ):
        if early_stopping:
            if not maximize and not minimize:
                print(
                    "[Warning]: early_stopping can be used only when maximizing or minimizing, the parameter will be ignored."
                )
                early_stopping = False
        self.early_stopping = early_stopping
        self.maximize = maximize
        self.minimize = minimize
        # Start from the worst possible value so that the first validation score is always an improvement
        self.best_value = -float("inf") if maximize else float("inf")
        self.tolerance = tolerance
        self.patience = patience
        self.current_patience = patience
        self.max_iterations = max_iterations
        self.warmup_iterations = warmup_iterations
        self.verbose = verbose
        self.__best_model_path = None
        self.__best_iteration = None

    def save_weights(self, model: nn.Module, iteration: int):
        self.delete_best_weights()
        # mkstemp gives a unique file, so concurrent runs never share a checkpoint
        fd, self.__best_model_path = tempfile.mkstemp(
            prefix=f"model_{iteration}_", suffix=".pyt"
        )
        os.close(fd)
        self.__best_iteration = iteration
        torch.save(model.state_dict(), self.__best_model_path)

    def load_best_weights(self, model: nn.Module):
        if not (self.__best_model_path is None):
            if self.verbose:
                print(
                    f"Loading the weights saved at iteration {self.__best_iteration}"
                )
            model.load_state_dict(torch.load(self.__best_model_path))
        else:
            if self.verbose:
                print(f"Using the weights from the last iteration")

    def delete_best_weights(self):
        if not (self.__best_model_path is None):
            if os.path.exists(self.__best_model_path):
                os.remove(self.__best_model_path)
            self.__best_model_path = None

    def should_stop(
        self, iteration: int, score: Optional[float], model: nn.Module
    ) -> bool:
        stop = False

        if iteration <= self.warmup_iterations:
            return False

        if self.early_stopping:
            improvement = (
                (score - self.best_value)
                if self.maximize
                else (self.best_value - score)
            )
            if improvement > 0:
                # Improvement, update best and reset the patience
                self.best_value = score
                self.current_patience = self.patience

                # Write the best state_dict to disk
                self.save_weights(model, iteration)
            elif -improvement < self.tolerance:
                pass
            else:
                self.current_patience -= 1

            if self.current_patience <= 0:
                stop = True

        if self.max_iterations:
            if iteration >= self.max_iterations:
                stop = True
        return stop

    def __del__(self):
        self.delete_best_weights()
