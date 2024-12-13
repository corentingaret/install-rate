# install_rate main file

import logging
from typing import Annotated
from typer import Typer, Argument

import helpers.logger
import helpers.download
import helpers.prepare
import helpers.save
import helpers.evaluate

import models


app = Typer(name="install_rate")


@app.command()
def prepare(
    limit: Annotated[int, Argument(envvar="LIMIT")],
    exec_date: Annotated[str, Argument(envvar="EXEC_DATE")],
    val_ratio: Annotated[float, Argument(envvar="VAL_RATIO")],
    minority_class_ratio: Annotated[float, Argument(envvar="MINORITY_CLASS_RATIO")],
) -> None:
    """
    Prepares the data for training and testing.

    Args:
        limit (int): The maximum number of rows to download.
        exec_date (str): The execution date.
        val_ratio (float): The ratio of the training data to be used for validation.
        minority_class_ratio (float): The ratio of the minority class to be used for balancing the dataset.
    """

    helpers.logger.init()

    df_train = helpers.download.get_data(path="data/origin/train.parquet", limit=limit)
    df_test = helpers.download.get_data(path="data/origin/test.parquet", limit=limit)

    logging.info(f"Preprocessing data")
    df_train, df_val, df_test = helpers.prepare.preprocessing(
        df_train=df_train,
        df_test=df_test,
        val_ratio=val_ratio,
        minority_class_ratio=minority_class_ratio,
    )

    helpers.save.save_data(name="train", data=df_train, exec_date=exec_date)
    helpers.save.save_data(name="val", data=df_val, exec_date=exec_date)
    helpers.save.save_data(name="test", data=df_test, exec_date=exec_date)

    return


@app.command()
def train(
    limit: Annotated[int, Argument(envvar="LIMIT")],
    exec_date: Annotated[str, Argument(envvar="EXEC_DATE")],
    model_ref: Annotated[str, Argument(envvar="MODEL_REF")],
    grid_search: Annotated[bool, Argument(envvar="GRID_SEARCH")],
) -> None:
    """
    Trains the model.
    If you created a new model in the models module, you need to add it to the __init__.py get_model function.

    Args:
        limit (int): The maximum number of rows to download.
        exec_date (str): The execution date.
        model_ref (str): The reference of the model.
        grid_search (bool): Whether to perform grid search.
    """

    helpers.logger.init()

    instance = models.get_model(model_ref=model_ref)

    df_train = helpers.download.get_data(
        path=f"data/{exec_date}/train.parquet", limit=limit
    )

    df_val = helpers.download.get_data(
        path=f"data/{exec_date}/val.parquet", limit=limit
    )

    instance.train_model(df_train=df_train, df_val=df_val, grid_search=grid_search)
    instance.save_model(exec_date=exec_date, model_ref=model_ref)

    return


@app.command()
def evaluate(
    limit: Annotated[int, Argument(envvar="LIMIT")],
    exec_date: Annotated[str, Argument(envvar="EXEC_DATE")],
    model_ref: Annotated[str, Argument(envvar="MODEL_REF")],
) -> None:
    """
    Evaluates the model.

    Args:
        limit (int): The maximum number of rows to download.
        exec_date (str): The execution date.
        model_ref (str): The reference of the model.
    """

    helpers.logger.init()

    df_test = helpers.download.get_data(
        path=f"data/{exec_date}/test.parquet", limit=limit
    )

    instance = models.get_model(model_ref=model_ref)
    model = instance.load_trained_model(exec_date=exec_date, model_ref=model_ref)

    helpers.evaluate.get_evaluation(df_test=df_test, model=model)

    return


if __name__ == "__main__":
    app()
