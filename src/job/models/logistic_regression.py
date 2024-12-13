import logging
import pandas as pd
from models.model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss


class LogisticRegressionModel(Model):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(solver="lbfgs", max_iter=100, C=0.01)

    def get_model(self) -> LogisticRegression:
        return self.model

    def train_model(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, grid_search: bool = False
    ) -> LogisticRegression:
        logging.info(f"data size: train={len(df_train)}, val={len(df_val)}")

        X_train = df_train.drop(columns="install_label")
        y_train = df_train["install_label"]
        X_val = df_val.drop(columns="install_label")
        y_val = df_val["install_label"]

        if grid_search:
            # Define the parameter grid
            param_grid = {
                "C": [0.01, 0.1, 1, 10, 100],
                "max_iter": [100, 200, 500, 1000],
            }

            # Initialize grid search
            grid_search = GridSearchCV(
                estimator=self.model, param_grid=param_grid, cv=3, n_jobs=-1
            )

            # Fit the model
            logging.info("Starting grid search for best parameters")
            grid_search.fit(X_train, y_train)

            # Update the model with the best estimator
            self.model = grid_search.best_estimator_
            logging.info(f"Best parameters found: {grid_search.best_params_}")

        else:
            # Train the model without grid search
            logging.info("Training model without grid search")
            self.model.fit(X_train, y_train)

        # Evaluate on validation
        val_proba = self.model.predict_proba(X_val)
        val_log_loss = log_loss(y_val, val_proba)
        logging.info(f"Validation log loss with best parameters: {val_log_loss}")

        return self.model

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict_proba(X_test)
