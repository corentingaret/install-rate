import pandas
import logging
import os
import joblib
from typing import Any


def save_data(name: str, data: pandas.DataFrame, exec_date: str) -> None:
    """
    Saves the data to a parquet file.

    Args:
        name (str): The name of the data to save.
        data (pandas.DataFrame): The data to save.
        exec_date (str): The execution date.
    """

    directory = f"data/{exec_date}"

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    logging.info(f"Saving {name} data to {directory}/{name}.parquet")
    data.to_parquet(f"{directory}/{name}.parquet")
    return


def save_model(model: Any, exec_date: str, model_ref: str) -> None:
    """
    Saves the model to a joblib file.

    Args:
        model (Any): The model to save.
        exec_date (str): The execution date.
        model_ref (str): The reference of the model.
    """

    directory = f"data/{exec_date}/{model_ref}"

    if not os.path.exists(directory):
        os.makedirs(directory)

    logging.info(f"Saving model to {directory}/model.joblib")
    joblib.dump(model, f"{directory}/model.joblib")
    return
