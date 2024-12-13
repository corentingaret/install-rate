from abc import ABC, abstractmethod
from typing import Any
import joblib
import os
import logging


class Model(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def get_model(self) -> Any:
        """Initialize and return the model instance."""
        pass

    @abstractmethod
    def train_model(self, data: Any) -> Any:
        """Train the model with the provided data."""
        pass

    @abstractmethod
    def predict_proba(self, data: Any) -> Any:
        """Predict the model with the provided data."""
        pass

    def save_model(self, exec_date: str, model_ref: str) -> None:
        """Save the model to the specified path."""
        if self.model is None:
            raise ValueError("Model is not initialized or trained.")

        path = f"data/{exec_date}/{model_ref}/model.joblib"

        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        logging.info(f"Saving model to {path}")
        joblib.dump(self.model, path)

    def load_trained_model(self, exec_date: str, model_ref: str) -> Any:
        """Load the model from the specified path."""

        path = f"data/{exec_date}/{model_ref}/model.joblib"

        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")

        logging.info(f"Loading model from {path}")
        self.model = joblib.load(path)
        return self.model
