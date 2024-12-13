import logging
import pandas
from sklearn.metrics import log_loss
from typing import Any


def get_evaluation(df_test: pandas.DataFrame, model: Any) -> None:
    X_test = df_test.drop(columns="install_label")
    y_test = df_test["install_label"]
    y_pred = model.predict(X_test)

    logging.info(f"Log Loss: {log_loss(y_test, y_pred)}")

    return
