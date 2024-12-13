"""models helpers"""

import logging

from models import logistic_regression


def get_model(model_ref: str):
    logging.info(f"loading model {model_ref}")
    return {
        "logistic_regression": logistic_regression.LogisticRegressionModel(),
    }[model_ref]
