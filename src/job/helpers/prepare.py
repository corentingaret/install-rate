import pandas
import logging
from itertools import chain
from sklearn.preprocessing import LabelEncoder, RobustScaler

import helpers.download
import helpers.feature_eng


def handle_missing_values(
    data: pandas.DataFrame, training_features: dict
) -> pandas.DataFrame:
    """
    Handle missing values in the data.

    Args:
        data (pandas.DataFrame): The input data containing features and target.
        training_features (dict): The dictionary containing the training features.

    Returns:
        pandas.DataFrame: The data without missing values and without dropped features.
    """
    for feature in chain.from_iterable(training_features.values()):
        if data[feature].isna().sum() / len(data) > 0.6:
            logging.info(
                f"Dropping feature {feature} because it has more than 60% missing values"
            )
            data = data.drop(columns=[feature])
            continue

        # Impute missing values
        if data[feature].dtype in ["float64", "int64"]:
            # Impute numerical features with median => data is skewed so similar to fillna(0)
            median_value = data[feature].median()
            data[feature].fillna(median_value, inplace=True)
        else:
            # Impute categorical features with mode
            mode_value = data[feature].mode()[0]
            data[feature].fillna(mode_value, inplace=True)

    return data


def common_preprocessing(
    data: pandas.DataFrame, training_features: dict
) -> pandas.DataFrame:
    """
    Common preprocessing steps for both training and test data.

    Args:
        data (pandas.DataFrame): The input data containing features and target.
        training_features (dict): The dictionary containing the training features.

    Returns:
        pandas.DataFrame: The preprocessed data.
    """

    # Make an explicit copy of the input DataFrame
    data = data.copy()

    # Drop duplicates
    data = data.drop_duplicates(subset=["user_id", "app", "bid_id", "bid_timestamp"])

    # Manage temporal
    temporal_features = [
        "bid_timestamp",
        "install_date",
        "session_start_date",
        "previous_session_start_date",
    ]

    # For datetime conversions, use proper assignment
    for feature in temporal_features:
        data.loc[:, feature] = pandas.to_datetime(data[feature])

    # Include feature engineering
    if "feature_engineering" in training_features:
        data = helpers.feature_eng.feature_engineering(
            data, training_features["feature_engineering"]
        )

    # Handle missing values
    data = handle_missing_values(data=data, training_features=training_features)

    # Limit to chosen features and target
    features = list(chain.from_iterable(training_features.values()))
    data = data[features + ["install_label"]]
    return data


def encode_categorical_features(
    df_train: pandas.DataFrame,
    df_test: pandas.DataFrame,
    df_val: pandas.DataFrame,
    training_features: dict,
) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    Encodes categorical features in the training and test datasets using one-hot encoding.

    Args:
        df_train (pandas.DataFrame): The training data containing features and target.
        df_test (pandas.DataFrame): The test data containing features and target.
        df_val (pandas.DataFrame): The validation data containing features and target.
        training_features (dict): The dictionary containing the training features.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: The encoded training and test datasets.
    """

    # Concatenate train and test data for consistent encoding
    combined_data = pandas.concat(
        [df_train, df_test, df_val], keys=["train", "test", "val"]
    )
    categorical_features = training_features["categorical"]

    # Initialize LabelEncoder
    encoder = LabelEncoder()

    # Apply label encoding to each categorical feature
    for feature in categorical_features:
        logging.info(f"Encoding feature {feature}")
        combined_data[feature] = encoder.fit_transform(combined_data[feature])

    # Split back into train and test sets
    X_train = combined_data.loc["train"]
    X_test = combined_data.loc["test"]
    X_val = combined_data.loc["val"]
    return X_train, X_test, X_val


def scale_numerical_features(
    df_train: pandas.DataFrame, df_test: pandas.DataFrame, df_val: pandas.DataFrame
) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    Scales the numerical features in the training and test datasets.

    Args:
        df_train (pandas.DataFrame): The training data containing features and target.
        df_test (pandas.DataFrame): The test data containing features and target.
        df_val (pandas.DataFrame): The validation data containing features and target.
    """
    scaler = RobustScaler()

    for feature in df_train.columns:
        if df_train[feature].dtype in ["float64", "int64"]:
            logging.info(f"Scaling feature {feature}")
            # Reshape the data to 2D before scaling
            df_train[feature] = scaler.fit_transform(
                df_train[feature].values.reshape(-1, 1)
            )
            df_test[feature] = scaler.transform(df_test[feature].values.reshape(-1, 1))
            df_val[feature] = scaler.transform(df_val[feature].values.reshape(-1, 1))
    return df_train, df_test, df_val


def balance_dataset(
    data: pandas.DataFrame, minority_class_ratio: float
) -> pandas.DataFrame:
    """
    Balances the dataset by undersampling the majority class.

    Args:
        data (pandas.DataFrame): The input data containing features and target.
        minority_class_ratio (float): The ratio of the minority class to be used for balancing the dataset.

    Returns:
        pandas.DataFrame: The balanced dataset.
    """

    majority_class = 0
    minority_class = 1

    # Separate the majority and minority classes
    majority_class_data = data[data["install_label"] == majority_class]
    minority_class_data = data[data["install_label"] == minority_class]

    # Undersample the majority class
    n_minority = len(minority_class_data)
    n_majority = int(n_minority * (1 - minority_class_ratio) / minority_class_ratio)

    # Check that we are not oversampling the majority class
    if n_majority > len(majority_class_data):
        n_majority = len(majority_class_data)

    majority_class_data_undersampled = majority_class_data.sample(
        n=n_majority, random_state=42
    )

    # Concatenate the undersampled majority class data with the minority class data
    balanced_data = pandas.concat(
        [majority_class_data_undersampled, minority_class_data]
    )

    return balanced_data


def split_data(
    data: pandas.DataFrame, val_ratio: float
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Splits the data into training and validation sets while avoiding temporal leakage.
    We ensure that the validation set is in the future of the training set.

    Args:
        data (pandas.DataFrame): The input data containing features and target.
        val_ratio (float): The ratio of the data to be used for validation.

    Returns:
        pandas.DataFrame: The training and validation datasets.
    """

    # Sort the data by timestamp
    data = data.sort_values(by="bid_timestamp")

    # Calculate the number of validation samples
    val_size = int(len(data) * val_ratio)

    # Split the data
    train_data = data[:-val_size]
    val_data = data[-val_size:]

    return train_data, val_data


def preprocessing(
    df_train: pandas.DataFrame,
    df_test: pandas.DataFrame,
    val_ratio: float,
    minority_class_ratio: float,
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Preprocesses the training and test data, encodes categorical features, and splits the training data into training and validation sets.

    Args:
        df_train (pandas.DataFrame): The training data containing features and target.
        df_test (pandas.DataFrame): The test data containing features and target.
        val_ratio (float): The ratio of the training data to be used for validation.
        minority_class_ratio (float): The ratio of the minority class to be used for balancing the dataset.
    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]: The processed training, validation, and test datasets.
    """

    training_features = helpers.download.get_features_from_yml("training_features.yaml")

    df_train, df_val = split_data(data=df_train, val_ratio=val_ratio)

    df_train = common_preprocessing(data=df_train, training_features=training_features)
    df_val = common_preprocessing(data=df_val, training_features=training_features)
    df_test = common_preprocessing(data=df_test, training_features=training_features)

    # Encoding and scaling
    df_train, df_test, df_val = scale_numerical_features(
        df_train=df_train, df_test=df_test, df_val=df_val
    )

    df_train, df_test, df_val = encode_categorical_features(
        df_train=df_train,
        df_test=df_test,
        df_val=df_val,
        training_features=training_features,
    )

    # Balancing the dataset
    df_train = balance_dataset(df_train, minority_class_ratio)

    return df_train, df_val, df_test
